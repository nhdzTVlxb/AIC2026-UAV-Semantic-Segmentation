from __future__ import annotations
from landcover.cli import command_arguments
import argparse
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path
import numpy as np
from landcover.commands._write_teacher_batches_support import image_sequence, normalized_teacher_forward, teacher_single_view, teacher_view_average, read_existing_teacher
from landcover.bootstrap import ROOT, prepare_runtime
from landcover.artifacts import file_digest, write_record
prepare_runtime()

def execute() -> None:
    checkout = ROOT
    arguments, options = command_arguments('write_teacher_batches')
    if options.batch_size < 1:
        raise ValueError('batch-size must be positive')
    segmentation_root = checkout / 'vendor' / 'InternImage' / 'segmentation'
    sys.path.insert(0, str(checkout))
    sys.path.insert(0, str(segmentation_root))
    os.environ.setdefault('AIC_PROJECT_ROOT', str(checkout))
    from landcover.checkpoints import allow_project_weights
    allow_project_weights()
    import mmcv_custom
    import mmseg_custom
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmseg.apis import init_segmentor
    from mmseg.apis.inference import LoadImage
    from mmseg.datasets.pipelines import Compose
    import torch
    images = sorted(options.test_dir.glob('test2_*.png'), key=image_sequence)
    expected = [f'test2_{index}.png' for index in range(1, options.expected_count + 1)]
    if [location.name for location in images] != expected:
        raise RuntimeError(f'Test2 names/count invalid: {len(images)}')
    if options.limit is not None:
        if options.limit <= 0:
            raise ValueError('limit must be positive')
        images = images[:options.limit]
        options.expected_count = options.limit
    if not options.checkpoint.is_file():
        raise FileNotFoundError(options.checkpoint)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    recipe = Config.fromfile(str(options.config))
    pipeline = Compose([LoadImage()] + recipe.data.test.pipeline[1:])
    model = init_segmentor(str(options.config), str(options.checkpoint), device=options.device)
    model.inference = types.MethodType(normalized_teacher_forward, model)
    model.simple_test = types.MethodType(teacher_single_view, model)
    model.aug_test = types.MethodType(teacher_view_average, model)
    device = next(model.parameters()).device
    pending = []
    for image_path in images:
        output_path = options.output_dir / f'{image_path.stem}.npz'
        cached = read_existing_teacher(output_path) if output_path.is_file() else None
        if cached is None:
            if output_path.is_file():
                output_path.unlink()
            pending.append(image_path)
    print(f'pending={len(pending)} cached={len(images) - len(pending)} batch_size={options.batch_size} barren_multiplier={options.barren_multiplier}', flush=True)
    start_time = time.time()
    generated = 0
    with torch.inference_mode():
        for start in range(0, len(pending), options.batch_size):
            batch_paths = pending[start:start + options.batch_size]
            batch_items = [pipeline(dict(img=str(location))) for location in batch_paths]
            view_counts = {len(item['img']) for item in batch_items}
            if view_counts != {6}:
                raise RuntimeError(f'expected six raw views, got {view_counts}')
            data = collate(batch_items, samples_per_gpu=len(batch_items))
            data = scatter(data, [device])[0]
            sample = model(return_loss=False, rescale=True, **data)
            if len(sample) != len(batch_paths):
                raise RuntimeError(f'model returned {len(sample)} results for {len(batch_paths)} images')
            for image_path, probability in zip(batch_paths, sample):
                probability = np.asarray(probability, dtype=np.float32)
                if probability.shape != (8, 1024, 1024):
                    raise RuntimeError(f'bad probability shape for {image_path.name}: {probability.shape}')
                probability[4] *= options.barren_multiplier
                probability /= np.maximum(probability.sum(axis=0, keepdims=True), 1e-08)
                q8 = np.rint(probability * 255.0).clip(0, 255).astype(np.uint8)
                output_path = options.output_dir / f'{image_path.stem}.npz'
                temporary = output_path.with_suffix('.npz.tmp')
                with temporary.open('wb') as handle:
                    np.savez(handle, probs_q8=q8)
                temporary.replace(output_path)
                generated += 1
            done = len(images) - len(pending) + min(start + len(batch_paths), len(pending))
            seconds = time.time() - start_time
            remaining = len(pending) - (done - (len(images) - len(pending)))
            eta = seconds / max(start + len(batch_paths), 1) * remaining
            print(f'{done}/{len(images)} generated={generated} elapsed={seconds / 60:.1f}min eta={eta / 60:.1f}min', flush=True)
    class_pixels = np.zeros(8, dtype=np.int64)
    confidence_histogram = np.zeros(256, dtype=np.int64)
    for image_path in images:
        q8 = read_existing_teacher(options.output_dir / f'{image_path.stem}.npz')
        if q8 is None:
            raise RuntimeError(f'invalid cached teacher file: {image_path.name}')
        hard = q8.argmax(axis=0)
        certainty = q8.max(axis=0)
        class_pixels += np.bincount(hard.reshape(-1), minlength=8)[:8]
        confidence_histogram += np.bincount(certainty.reshape(-1), minlength=256)[:256]
    total = int(confidence_histogram.sum())
    record = {'config': str(options.config.resolve()), 'config_sha256': file_digest(options.config), 'checkpoint': str(options.checkpoint.resolve()), 'checkpoint_sha256': file_digest(options.checkpoint), 'test_dir': str(options.test_dir.resolve()), 'image_count': len(images), 'view_protocol': 'scales=[0.75,1.0,1.125], identity+horizontal, whole', 'batch_size': options.batch_size, 'barren_multiplier': options.barren_multiplier, 'quantization': 'uint8 round(probability*255), renormalize on load', 'class_pixel_counts_0_to_7': class_pixels.tolist(), 'confidence_q8_histogram': confidence_histogram.tolist(), 'fraction_ge_090': float(confidence_histogram[230:].sum() / total), 'fraction_050_to_090': float(confidence_histogram[128:230].sum() / total), 'total_seconds': round(time.time() - start_time, 3)}
    report_path = options.output_dir.parent / 'teacher_report.json'
    report_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'report: {report_path}')
if __name__ == '__main__':
    execute()
