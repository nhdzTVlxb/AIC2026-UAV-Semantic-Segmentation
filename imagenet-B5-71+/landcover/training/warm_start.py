"""Load matching official tensors and check checkpoint coverage."""
from pathlib import Path
from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class LoadMatchingWeights(Hook):
    """Load compatible tensors and explicitly skip shape-mismatched heads."""

    def __init__(self, checkpoint, min_loaded_ratio=0.95):
        self.checkpoint = checkpoint
        self.min_loaded_ratio = float(min_loaded_ratio)

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, 'module') else model

    def before_run(self, runner):
        import torch
        from mmcv.runner.checkpoint import _load_checkpoint
        weight_file = Path(self.checkpoint)
        if not weight_file.is_file():
            raise FileNotFoundError(weight_file)
        weight_record = _load_checkpoint(str(weight_file), map_location='cpu', logger=runner.logger)
        if isinstance(weight_record, dict) and 'state_dict' in weight_record:
            incoming = weight_record['state_dict']
        elif isinstance(weight_record, dict) and 'model' in weight_record:
            incoming = weight_record['model']
        else:
            incoming = weight_record
        incoming = {key[7:] if key.startswith('module.') else key: value for key, value in incoming.items()}
        model = self._unwrap(runner.model)
        expected = model.state_dict()
        matched, excluded = ({}, {})
        for key, value in incoming.items():
            if key not in expected:
                excluded[key] = {'reason': 'missing_in_target', 'source_shape': list(value.shape)}
            elif tuple(value.shape) != tuple(expected[key].shape):
                excluded[key] = {'reason': 'shape_mismatch', 'source_shape': list(value.shape), 'target_shape': list(expected[key].shape)}
            else:
                matched[key] = value
        fraction = len(matched) / max(len(expected), 1)
        if fraction < self.min_loaded_ratio:
            raise RuntimeError(f'compatible checkpoint ratio too low: {len(matched)}/{len(expected)}={fraction:.3f}')
        incompatible = model.load_state_dict(matched, strict=False)
        runner.logger.info('Loaded compatible checkpoint %s: %d/%d target tensors (%.2f%%); skipped=%s; missing_after_filter=%s; unexpected=%s', weight_file, len(matched), len(expected), fraction * 100.0, excluded, incompatible.missing_keys, incompatible.unexpected_keys)
