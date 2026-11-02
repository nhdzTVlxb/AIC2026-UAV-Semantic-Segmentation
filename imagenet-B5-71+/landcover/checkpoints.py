"""Checkpoint loading compatibility for the project's trusted weights."""
import inspect
from functools import wraps


def allow_project_weights():
    import torch
    loader = torch.load
    if "weights_only" not in inspect.signature(loader).parameters:
        return False
    if getattr(loader, "_aic_trusted_compat", False):
        return True

    @wraps(loader)
    def load_with_legacy_default(*positional, **options):
        options.setdefault("weights_only", False)
        return loader(*positional, **options)

    load_with_legacy_default._aic_trusted_compat = True
    torch.load = load_with_legacy_default
    return True
