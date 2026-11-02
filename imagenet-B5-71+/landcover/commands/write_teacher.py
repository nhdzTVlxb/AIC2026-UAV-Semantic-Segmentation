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
from landcover.commands._write_teacher_support import image_sequence, normalized_teacher_forward, teacher_single_view, teacher_view_average
from landcover.bootstrap import ROOT, prepare_runtime
from landcover.artifacts import file_digest, write_record
prepare_runtime()

def execute() -> None:
    checkout = ROOT
    arguments, options = command_arguments('write_teacher')
    segmentation_root = checkout / 'vendor' / 'InternImage' / 'segmentation'
    sys.path.insert(0, str(checkout))
    sys.path.insert(0, str(segmentation_root))
    os.environ.setdefault('AIC_PROJECT_ROOT', str(checkout))
    from landcover.checkpoints import allow_project_weights
    allow_project_weights()
    import mmcv_custom
    import mmseg_custom
    from mmseg.apis import inference_segmentor, init_segmentor
    images = sorted(options.test_dir.glob('test2_*.png'), key=image_sequence)
    if options.limit is not None:
        images = images[:options.limit]
        options.expected_count = options.limit
    expected = [f'test2_{index}.png' for index in range(1, options.expected_count + 1)]
    if [location.name for location in images] != expected:
        raise RuntimeError(f'Test2 names/count invalid: {len(images)}')
    if not options.checkpoint.is_file():
        raise FileNotFoundError(options.checkpoint)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    model = init_segmentor(str(options.config), str(options.checkpoint), device=options.device)
    model.inference = types.MethodType(normalized_teacher_forward, model)
    model.simple_test = types.MethodType(teacher_single_view, model)
    model.aug_test = types.MethodType(teacher_view_average, model)
    class_pixels = np.zeros(8, dtype=np.int64)
    confidence_histogram = np.zeros(256, dtype=np.int64)
    start_time = time.time()
    generated = 0
    for index, image_path in enumerate(images, 1):
        output_path = options.output_dir / f'{image_path.stem}.npz'
        if output_path.is_file():
            with np.load(output_path) as payload:
                q8 = payload['probs_q8']
        else:
            probability = np.asarray(inference_segmentor(model, str(image_path))[0], dtype=np.float32)
            if probability.shape != (8, 1024, 1024):
                raise RuntimeError(f'bad probability shape for {image_path.name}: {probability.shape}')
            probability[4] *= options.barren_multiplier
            probability /= np.maximum(probability.sum(axis=0, keepdims=True), 1e-08)
            q8 = np.rint(probability * 255.0).clip(0, 255).astype(np.uint8)
            np.savez(output_path, probs_q8=q8)
            generated += 1
        if q8.shape != (8, 1024, 1024) or q8.dtype != np.uint8:
            raise RuntimeError(f'invalid cached teacher file: {output_path}')
        hard = q8.argmax(axis=0)
        certainty = q8.max(axis=0)
        class_pixels += np.bincount(hard.reshape(-1), minlength=8)[:8]
        confidence_histogram += np.bincount(certainty.reshape(-1), minlength=256)[:256]
        if index % 10 == 0 or index == len(images):
            seconds = time.time() - start_time
            eta = seconds / index * (len(images) - index)
            print(f'{index}/{len(images)} generated={generated} elapsed={seconds / 60:.1f}min eta={eta / 60:.1f}min', flush=True)
    total = int(confidence_histogram.sum())
    record = {'config': str(options.config.resolve()), 'config_sha256': file_digest(options.config), 'checkpoint': str(options.checkpoint.resolve()), 'checkpoint_sha256': file_digest(options.checkpoint), 'test_dir': str(options.test_dir.resolve()), 'image_count': len(images), 'view_protocol': 'scales=[0.75,1.0,1.125], identity+horizontal, whole', 'barren_multiplier': options.barren_multiplier, 'quantization': 'uint8 round(probability*255), renormalize on load', 'class_pixel_counts_0_to_7': class_pixels.tolist(), 'confidence_q8_histogram': confidence_histogram.tolist(), 'fraction_ge_090': float(confidence_histogram[230:].sum() / total), 'fraction_050_to_090': float(confidence_histogram[128:230].sum() / total), 'total_seconds': round(time.time() - start_time, 3)}
    report_path = options.output_dir.parent / 'teacher_report.json'
    report_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'report: {report_path}')
if __name__ == '__main__':
    execute()
