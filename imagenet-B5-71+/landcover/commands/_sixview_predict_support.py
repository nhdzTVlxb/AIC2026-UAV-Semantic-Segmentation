import argparse
import gc
import hashlib
import json
import os
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from landcover.bootstrap import ROOT, prepare_runtime
from landcover.artifacts import file_digest, write_record
prepare_runtime()
PROJECT = ROOT

class SixViewImages(Dataset):

    def __init__(self, paths, config):
        self.paths, self.config, self.pipeline = (paths, config, None)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.pipeline is None:
            import cv2
            cv2.setNumThreads(1)
            torch.set_num_threads(1)
            from mmcv import Config
            from mmseg.apis.inference import LoadImage
            from mmseg.datasets.pipelines import Compose
            recipe = Config.fromfile(self.config)
            self.pipeline = Compose([LoadImage()] + recipe.data.test.pipeline[1:])
        item = self.pipeline(dict(img=str(self.paths[index])))
        if len(item['img']) != 6:
            raise RuntimeError('P1 must have exactly six views')
        return dict(img=item['img'], metas=[meta.data for meta in item['img_metas']], path=str(self.paths[index]))

def stack_views(items):
    return dict(img=[torch.stack([item['img'][v] for item in items]) for v in range(6)], metas=[[item['metas'][v] for item in items] for v in range(6)], paths=[item['path'] for item in items])

def create_loader(paths, config, workers):
    return DataLoader(SixViewImages(paths, config), batch_size=4, num_workers=workers, collate_fn=stack_views, pin_memory=True, prefetch_factor=1, multiprocessing_context='spawn', persistent_workers=False)

def close_loader(iterator):
    if hasattr(iterator, '_shutdown_workers'):
        iterator._shutdown_workers()

def transfer_views(parts):
    image_paths = sum([loader_part['paths'] for loader_part in parts], [])
    metas = [sum([loader_part['metas'][v] for loader_part in parts], []) for v in range(6)]
    for view in metas:
        if len({tuple(m['ori_shape']) for m in view}) != 1:
            raise RuntimeError('Mixed image shapes cannot share the existing model batch')
    images = [torch.cat([loader_part['img'][v].to('cuda', non_blocking=True) for loader_part in parts]) for v in range(6)]
    return dict(img=images, metas=metas, paths=image_paths)

@torch.inference_mode()
def average_views(model, batch):
    result = model.inference(batch['img'][0], batch['metas'][0], True)
    for v in range(1, 6):
        result += model.inference(batch['img'][v], batch['metas'][v], True)
    return result / 6

def configure_attention(model, enabled):
    quantity = 0
    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            if not hasattr(module, '_aic_original_forward'):
                module._aic_original_forward = module.forward
            if enabled:

                def forward(self, *args, **kwargs):
                    kwargs['need_weights'] = False
                    return self._aic_original_forward(*args, **kwargs)
                module.forward = types.MethodType(forward, module)
            else:
                module.forward = module._aic_original_forward
            quantity += 1
    return quantity

def store_mask(path, prediction, output):
    with Image.open(path) as image:
        shape = (image.height, image.width)
    if prediction.shape != shape or prediction.min() < 0 or prediction.max() > 7:
        raise RuntimeError(f'Invalid prediction: {path}')
    expected = output / Path(path).name
    temporary = expected.with_suffix('.png.tmp')
    Image.fromarray((prediction + 1).astype(np.uint8)).save(temporary, format='PNG', compress_level=3)
    temporary.replace(expected)
