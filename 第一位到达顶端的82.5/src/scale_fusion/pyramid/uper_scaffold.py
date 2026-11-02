import torch
from torch import nn
from torch.nn import functional as F
from scale_fusion.priors.mtp_binding import public_archives
from scale_fusion.pyramid.eight_class_row import VehicleExtension

class PyramidDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        public_archives()
        from mmcv.cnn import ConvModule

        def block(source, target, kernel, inplace=True):
            return ConvModule(source, target, kernel, padding=kernel // 2, norm_cfg={'type': 'SyncBN', 'requires_grad': True}, act_cfg={'type': 'ReLU'}, inplace=inplace)
        self.psp_modules = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(size), block(1536, 512, 1)) for size in (1, 2, 3, 6)])
        self.bottleneck = block(1536 + 4 * 512, 512, 3)
        self.lateral_convs = nn.ModuleList([block(channels, 512, 1, False) for channels in (192, 384, 768)])
        self.fpn_convs = nn.ModuleList([block(512, 512, 3, False) for _ in range(3)])
        self.fpn_bottleneck = block(4 * 512, 512, 3)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = VehicleExtension()

    @staticmethod
    def resize(value, shape):
        return F.interpolate(value, shape, mode='bilinear', align_corners=False)

    def forward(self, stages):
        pooled = [self.resize(branch(stages[-1]), stages[-1].shape[-2:]) for branch in self.psp_modules]
        outputs = [unit(stage) for unit, stage in zip(self.lateral_convs, stages[:3])]
        outputs.append(self.bottleneck(torch.cat((stages[-1], *pooled), dim=1)))
        for index in (2, 1, 0):
            outputs[index] = outputs[index] + self.resize(outputs[index + 1], outputs[index].shape[-2:])
        pyramid = [layer(value) for layer, value in zip(self.fpn_convs, outputs[:3])] + [outputs[-1]]
        shape = pyramid[0].shape[-2:]
        fused = self.fpn_bottleneck(torch.cat([self.resize(value, shape) for value in pyramid], dim=1))
        return (self.conv_seg(self.dropout(fused)), fused)
