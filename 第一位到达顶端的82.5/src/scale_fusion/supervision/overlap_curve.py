import torch

def lovasz(probabilities, labels):
    terms = []
    for c in range(8):
        truth = labels.eq(c).float()
        mass = truth.sum()
        if not mass:
            continue
        error, order = (truth - probabilities[:, c]).abs().sort(descending=True)
        ranked = truth[order]
        intersection = mass - ranked.cumsum(0)
        union = mass + (1 - ranked).cumsum(0)
        curve = 1 - intersection / union
        terms.append(torch.dot(error, torch.cat((curve[:1], curve[1:] - curve[:-1]))))
    return torch.stack(terms).mean() if terms else probabilities.sum() * 0
