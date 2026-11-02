import torch
from torch import nn
from torch.nn import functional as F
from scale_fusion.priors.mtp_binding import encoder_class, load_public_parameters
from scale_fusion.workspace_state import PUBLIC_SHA
from scale_fusion.pyramid.pixel_scale_gate import AdaptiveDecoder

class ScaleAdaptiveSegmenter(nn.Module):

    def __init__(self, checkpointing=True):
        super().__init__()
        self.backbone = encoder_class()(core_op='DCNv3_pytorch', channels=192, depths=[5, 5, 24, 5], groups=[12, 24, 48, 96], mlp_ratio=4.0, drop_path_rate=0.18, norm_layer='LN', layer_scale=1e-05, offset_scale=2.0, post_norm=True, with_cp=checkpointing, out_indices=(0, 1, 2, 3), pretrained=None)
        self.decode_head = AdaptiveDecoder()

    def initialize_public(self):
        parameters = load_public_parameters()
        missing = self.load_state_dict(parameters, strict=False)
        expected = {k for k in self.state_dict() if k.startswith('decode_head.scale_') or 'vehicle_' in k}
        if set(missing.missing_keys) != expected or missing.unexpected_keys:
            raise ValueError('Public state mismatch: ' + str(missing))
        if not all((torch.equal(self.state_dict()[k], v) for k, v in parameters.items())):
            raise ValueError('Original 949 public tensors changed')
        with torch.no_grad():
            c = self.decode_head.conv_seg
            c.vehicle_weight.copy_(c.weight[2:3])
            c.vehicle_bias.copy_(c.bias[2:3] - 2.7)
        return {'public_sha256': PUBLIC_SHA, 'retained_tensors': len(parameters), 'original_public_tensors_exact': True, 'vehicle_initialization': 'public road row; road bias minus 2.7'}

    def forward(self, normalized):
        coarse, auxiliary = self.decode_head(self.backbone(normalized))
        logits = F.interpolate(coarse.float(), normalized.shape[-2:], mode='bilinear', align_corners=False)
        return (logits, auxiliary)
