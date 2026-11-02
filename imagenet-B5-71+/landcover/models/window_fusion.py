"""Inference-only Mask2Former variant for the combined submission protocol."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from mmseg.models.builder import SEGMENTORS
from mmseg.ops import resize
from mmseg_custom.models.segmentors.encoder_decoder_mask2former import EncoderDecoderMask2Former

@SEGMENTORS.register_module()
class WindowFusionMask2Former(EncoderDecoderMask2Former):
    """Mask2Former with positive-Hann sliding fusion and class calibration."""

    def __init__(self, *args, barren_multiplier=1.0, hann_minimum=0.125, **kwargs):
        super().__init__(*args, **kwargs)
        self.barren_multiplier = float(barren_multiplier)
        self.hann_minimum = float(hann_minimum)
        if not 0.0 < self.hann_minimum <= 1.0:
            raise ValueError('hann_minimum must be in (0, 1]')

    def _positive_hann(self, height, width, *, device, dtype):
        wy = torch.hann_window(height, periodic=False, device=device, dtype=dtype)
        wx = torch.hann_window(width, periodic=False, device=device, dtype=dtype)
        window = wy[:, None] * wx[None, :]
        return self.hann_minimum + (1.0 - self.hann_minimum) * window

    def slide_inference(self, img, img_meta, rescale):
        step_y, step_x = self.test_cfg.stride
        crop_height, crop_width = self.test_cfg.crop_size
        batch_size, _, image_height, image_width = img.size()
        row_tiles = max(image_height - crop_height + step_y - 1, 0) // step_y + 1
        column_tiles = max(image_width - crop_width + step_x - 1, 0) // step_x + 1
        accumulated_scores = img.new_zeros((batch_size, self.num_classes, image_height, image_width))
        accumulated_weights = img.new_zeros((batch_size, 1, image_height, image_width))
        for row_index in range(row_tiles):
            for column_index in range(column_tiles):
                y1 = row_index * step_y
                x1 = column_index * step_x
                y2 = min(y1 + crop_height, image_height)
                x2 = min(x1 + crop_width, image_width)
                y1 = max(y2 - crop_height, 0)
                x1 = max(x2 - crop_width, 0)
                image_patch = img[:, :, y1:y2, x1:x2]
                patch_scores = self.encode_decode(image_patch, img_meta)
                weight = self._positive_hann(y2 - y1, x2 - x1, device=patch_scores.device, dtype=patch_scores.dtype)[None, None]
                weighted = patch_scores * weight
                padding = (int(x1), int(image_width - x2), int(y1), int(image_height - y2))
                accumulated_scores += F.pad(weighted, padding)
                accumulated_weights += F.pad(weight, padding)
        if torch.any(accumulated_weights <= 0):
            raise RuntimeError('positive-Hann sliding fusion left uncovered pixels')
        accumulated_scores = accumulated_scores / accumulated_weights
        if rescale:
            accumulated_scores = resize(accumulated_scores, size=img_meta[0]['ori_shape'][:2], mode='bilinear', align_corners=self.align_corners, warning=False)
        return accumulated_scores

    def _calibrated_argmax(self, probabilities):
        calibrated = probabilities.clone()
        calibrated[:, 4].mul_(self.barren_multiplier)
        return calibrated.argmax(dim=1).cpu().numpy()

    def simple_test(self, img, img_meta, rescale=True):
        distribution = self.inference(img, img_meta, rescale)
        return list(self._calibrated_argmax(distribution))

    def aug_test(self, imgs, img_metas, rescale=True):
        if not rescale:
            raise AssertionError('aug_test requires rescale=True')
        distribution = self.inference(imgs[0], img_metas[0], rescale)
        for index in range(1, len(imgs)):
            distribution += self.inference(imgs[index], img_metas[index], rescale)
        distribution /= len(imgs)
        return list(self._calibrated_argmax(distribution))
