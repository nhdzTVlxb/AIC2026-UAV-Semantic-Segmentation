import torch
from torch.nn import functional as F
from scale_fusion.supervision.overlap_curve import lovasz

def loss_terms(prediction, auxiliary, labels, alignment=False):
    active = labels.ne(255)
    if not active.any():
        zero = prediction.sum() * 0 + sum((v.sum() * 0 for v in auxiliary.values()))
        return (zero, {'ce': zero.detach(), 'lovasz': zero.detach(), 'scale_ce': zero.detach()})
    prediction = prediction.float()
    weights = prediction.new_tensor([1.1, 1, 1, 1, 1, 1, 1, 1])
    ce = F.cross_entropy(prediction, labels, weight=weights, ignore_index=255)
    if alignment:
        zero = ce.detach() * 0
        return (ce, {'ce': ce.detach(), 'lovasz': zero, 'scale_ce': zero})
    overlap = lovasz(prediction.softmax(1).movedim(1, -1)[active], labels[active])
    coarse = auxiliary['levels'].float()
    target = F.interpolate(labels[:, None].float(), size=coarse.shape[-2:], mode='nearest')[:, 0].long()
    scale_ce = torch.stack([F.cross_entropy(coarse[:, k], target, ignore_index=255) for k in range(4)]).mean() if target.ne(255).any() else coarse.sum() * 0
    total = ce + 0.5 * overlap + 0.12 * scale_ce
    return (total, {k: v.detach() for k, v in dict(ce=ce, lovasz=overlap, scale_ce=scale_ce).items()})
