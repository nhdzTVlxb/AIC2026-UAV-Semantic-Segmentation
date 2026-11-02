from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path
import numpy as np
from landcover.bootstrap import ROOT, prepare_runtime
from landcover.artifacts import file_digest, write_record
prepare_runtime()

def image_sequence(path: Path):
    suffix = path.stem.rsplit('_', 1)[-1]
    return int(suffix) if suffix.isdigit() else path.name

def normalized_teacher_forward(self, img, img_meta, rescale):
    if self.test_cfg.mode == 'slide':
        score = self.slide_inference(img, img_meta, rescale)
    else:
        score = self.whole_inference(img, img_meta, rescale)
    probability = score.float().clamp_min(1e-08)
    probability /= probability.sum(dim=1, keepdim=True).clamp_min(1e-08)
    if img_meta[0]['flip']:
        direction = img_meta[0]['flip_direction']
        if direction == 'horizontal':
            probability = probability.flip(dims=(3,))
        elif direction == 'vertical':
            probability = probability.flip(dims=(2,))
        else:
            raise ValueError(f'unsupported flip direction: {direction}')
    return probability

def teacher_single_view(self, img, img_meta, rescale=True):
    probability = self.inference(img, img_meta, rescale)
    return [item.detach().float().cpu().numpy() for item in probability]

def teacher_view_average(self, imgs, img_metas, rescale=True):
    if not rescale:
        raise AssertionError('probability teacher requires rescale=True')
    probability = self.inference(imgs[0], img_metas[0], rescale)
    for index in range(1, len(imgs)):
        probability += self.inference(imgs[index], img_metas[index], rescale)
    probability /= len(imgs)
    return [item.detach().float().cpu().numpy() for item in probability]

def read_existing_teacher(path: Path):
    try:
        with np.load(path) as payload:
            q8 = payload['probs_q8']
        if q8.shape != (8, 1024, 1024) or q8.dtype != np.uint8:
            return None
        if np.any(q8.sum(axis=0) == 0):
            return None
        return q8
    except (OSError, KeyError, ValueError):
        return None
