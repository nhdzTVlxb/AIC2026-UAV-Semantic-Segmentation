from __future__ import annotations
from landcover.cli import command_arguments
import argparse
import importlib
import json
import os
from pathlib import Path
from landcover.bootstrap import ROOT, prepare_runtime
from landcover.artifacts import file_digest, write_record
prepare_runtime()

def execute() -> None:
    arguments, options = command_arguments('score_holdout')
    checkout = ROOT
    official_code = checkout / 'vendor' / 'InternImage' / 'segmentation'
    import sys
    sys.path.insert(0, str(checkout))
    sys.path.insert(0, str(official_code))
    os.environ.setdefault('AIC_PROJECT_ROOT', str(checkout))
    from landcover.checkpoints import allow_project_weights
    allow_project_weights()
    import mmcv_custom
    import mmseg_custom
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmseg.apis import single_gpu_test
    from mmseg.datasets import build_dataloader, build_dataset
    from mmseg.models import build_segmentor
    recipe = Config.fromfile(str(options.config))
    for module in recipe.get('custom_imports', {}).get('imports', []):
        importlib.import_module(module)
    dataset_cfg = recipe.data.val.copy()
    dataset_cfg.test_mode = True
    dataset = build_dataset(dataset_cfg)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=recipe.data.get('workers_per_gpu', 2), dist=False, shuffle=False)
    recipe.model.pretrained = None
    recipe.model.train_cfg = None
    model = build_segmentor(recipe.model, test_cfg=recipe.get('test_cfg'))
    weight_record = load_checkpoint(model, str(options.checkpoint), map_location='cpu')
    model.CLASSES = weight_record.get('meta', {}).get('CLASSES', dataset.CLASSES)
    model.PALETTE = weight_record.get('meta', {}).get('PALETTE', dataset.PALETTE)
    model = MMDataParallel(model, device_ids=[0])
    torch.cuda.empty_cache()
    outputs = single_gpu_test(model, data_loader, pre_eval=True)
    metrics = {key: float(value) for key, value in dataset.evaluate(outputs, metric='mIoU').items()}
    result = {'config': str(options.config.resolve()), 'checkpoint': str(options.checkpoint.resolve()), 'split': str(dataset_cfg.split), 'sample_count': len(dataset), 'metrics': metrics, 'mIoU': float(metrics['mIoU'])}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
if __name__ == '__main__':
    execute()
