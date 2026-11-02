import torch
from torch import nn
from scale_fusion.pyramid.uper_scaffold import PyramidDecoder

class AdaptiveDecoder(PyramidDecoder):

    def __init__(self):
        super().__init__()
        self.scale_projects = nn.ModuleList([nn.Sequential(nn.Conv2d(512, 16, 1), nn.GELU()) for _ in range(4)])
        self.scale_predictor = nn.Conv2d(64, 4, 3, padding=1)
        nn.init.zeros_(self.scale_predictor.weight)
        nn.init.zeros_(self.scale_predictor.bias)

    def forward(self, stages):
        pooled = [self.resize(branch(stages[-1]), stages[-1].shape[-2:]) for branch in self.psp_modules]
        outputs = [layer(value) for layer, value in zip(self.lateral_convs, stages[:3])]
        outputs.append(self.bottleneck(torch.cat((stages[-1], *pooled), dim=1)))
        for k in (2, 1, 0):
            outputs[k] = outputs[k] + self.resize(outputs[k + 1], outputs[k].shape[-2:])
        pyramid = [layer(value) for layer, value in zip(self.fpn_convs, outputs[:3])] + [outputs[-1]]
        shape = pyramid[0].shape[-2:]
        aligned = [self.resize(value, shape) for value in pyramid]
        compressed = torch.cat([layer(value) for layer, value in zip(self.scale_projects, aligned)], dim=1)
        weights = self.scale_predictor(compressed).float().softmax(dim=1)
        modulated = [value * (4 * weights[:, k:k + 1].to(value.dtype)) for k, value in enumerate(aligned)]
        fused = self.fpn_bottleneck(torch.cat(modulated, dim=1))
        logits = self.conv_seg(self.dropout(fused))
        levels = torch.stack([self.conv_seg(value) for value in aligned], dim=1)
        return (logits, {'coarse': logits, 'levels': levels, 'weights': weights})
