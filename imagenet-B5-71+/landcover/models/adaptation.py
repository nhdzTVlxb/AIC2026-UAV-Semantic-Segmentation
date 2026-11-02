"""Target-domain self-training extension for AIC-2026 Mask2Former."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from mmseg.models.builder import SEGMENTORS
from mmseg_custom.models.segmentors.encoder_decoder_mask2former import EncoderDecoderMask2Former

@SEGMENTORS.register_module()
class TeacherGuidedMask2Former(EncoderDecoderMask2Former):
    """Use regular query losses on source and dense HK losses on Test2.

    A target sample is identified by the presence of ``teacher_probs``.  High
    confidence teacher pixels receive hard pseudo-label NLL.  The remaining
    moderately confident ("rejected") pixels retain the teacher distribution
    through forward KL.  Very uncertain pixels are ignored.
    """

    def __init__(self, *args, pseudo_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        pseudo_cfg = pseudo_cfg or {}
        accept_threshold = self._class_limits(pseudo_cfg.get('accept_threshold', 0.9), self.num_classes, 'accept_threshold')
        reject_floor = self._class_limits(pseudo_cfg.get('reject_floor', 0.5), self.num_classes, 'reject_floor')
        if torch.any(reject_floor >= accept_threshold):
            raise ValueError('every reject floor must be below its accept threshold')
        self.register_buffer('pseudo_accept_threshold', accept_threshold, persistent=False)
        self.register_buffer('pseudo_reject_floor', reject_floor, persistent=False)
        self.pseudo_hard_weight = float(pseudo_cfg.get('hard_weight', 0.5))
        self.pseudo_kl_weight = float(pseudo_cfg.get('kl_weight', 0.25))
        self.pseudo_hard_reliability = float(pseudo_cfg.get('hard_reliability', 0.5))
        self.pseudo_disagreement_weight = float(pseudo_cfg.get('disagreement_weight', 0.25))
        class_weight = pseudo_cfg.get('class_weight', [0.75, 0.85, 1.0, 1.0, 1.75, 0.75, 1.0, 1.25])
        if len(class_weight) != self.num_classes:
            raise ValueError('pseudo class_weight must match num_classes')
        self.register_buffer('pseudo_class_weight', torch.tensor(class_weight, dtype=torch.float32), persistent=False)
        if not 0.0 <= self.pseudo_hard_reliability <= 1.0:
            raise ValueError('hard_reliability must be within [0, 1]')
        if not 0.0 <= self.pseudo_disagreement_weight <= 1.0:
            raise ValueError('disagreement_weight must be within [0, 1]')

    @staticmethod
    def _class_limits(value, num_classes, name):
        class_limits = torch.as_tensor(value, dtype=torch.float32).flatten()
        if class_limits.numel() == 1:
            class_limits = class_limits.repeat(num_classes)
        if class_limits.numel() != num_classes:
            raise ValueError(f'{name} must be scalar or have one value per class')
        if torch.any(class_limits < 0.0) or torch.any(class_limits > 1.0):
            raise ValueError(f'{name} values must be within [0, 1]')
        return class_limits

    @staticmethod
    def _accepted_average(values, mask):
        mask = mask.to(values.dtype)
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _soft_average(values, mask, weights):
        pixel_weights = mask.to(values.dtype) * weights.to(values.dtype)
        return (values * pixel_weights).sum() / pixel_weights.sum().clamp_min(1.0)

    def _forward_target(self, img, img_metas, teacher_probs, teacher_reliability=None):
        return compute_teacher_losses(self, img, img_metas, teacher_probs, teacher_reliability)

    def forward_train(self, img, img_metas, gt_semantic_seg=None, teacher_probs=None, teacher_reliability=None, **kwargs):
        if teacher_probs is not None:
            return self._forward_target(img, img_metas, teacher_probs, teacher_reliability)
        if gt_semantic_seg is None:
            raise ValueError('source sample is missing gt_semantic_seg')
        return super().forward_train(img, img_metas, gt_semantic_seg, **kwargs)

def compute_teacher_losses(self, img, img_metas, teacher_probs, teacher_reliability=None):
    feature_maps = self.extract_feat(img)
    dense_scores = self.decode_head.forward_test(feature_maps, img_metas, self.test_cfg)
    student_distribution = dense_scores.float().clamp_min(1e-06)
    student_distribution = student_distribution / student_distribution.sum(dim=1, keepdim=True).clamp_min(1e-06)
    teacher_distribution = F.interpolate(teacher_probs.float(), size=student_distribution.shape[-2:], mode='bilinear', align_corners=False).clamp_min(1e-06)
    teacher_distribution = teacher_distribution / teacher_distribution.sum(dim=1, keepdim=True).clamp_min(1e-06)
    certainty, winning_class = teacher_distribution.max(dim=1)
    accept_threshold = self.pseudo_accept_threshold[winning_class]
    reject_floor = self.pseudo_reject_floor[winning_class]
    if teacher_reliability is None:
        agreement = torch.ones_like(certainty)
        hard_region = certainty >= accept_threshold
        soft_region = (certainty >= reject_floor) & (certainty < accept_threshold)
        kl_pixel_weight = agreement
    else:
        agreement = F.interpolate(teacher_reliability.float(), size=student_distribution.shape[-2:], mode='nearest').squeeze(1).clamp(0.0, 1.0)
        hard_region = (certainty >= accept_threshold) & (agreement >= self.pseudo_hard_reliability)
        soft_region = (certainty >= reject_floor) & ~hard_region
        kl_pixel_weight = self.pseudo_disagreement_weight + (1.0 - self.pseudo_disagreement_weight) * agreement
    student_log_prob = student_distribution.log()
    class_nll = -student_log_prob.gather(1, winning_class.unsqueeze(1)).squeeze(1)
    class_nll = class_nll * self.pseudo_class_weight[winning_class]
    accepted_loss = self._accepted_average(class_nll, hard_region)
    divergence = (teacher_distribution * (teacher_distribution.log() - student_log_prob)).sum(dim=1)
    soft_loss = self._soft_average(divergence, soft_region, kl_pixel_weight)
    return {'loss_pseudo_hard': accepted_loss * self.pseudo_hard_weight, 'loss_pseudo_rejected_kl': soft_loss * self.pseudo_kl_weight, 'pseudo_accept_ratio': hard_region.float().mean(), 'pseudo_reject_ratio': soft_region.float().mean(), 'pseudo_teacher_conf': certainty.mean(), 'pseudo_teacher_reliability': agreement.mean()}
