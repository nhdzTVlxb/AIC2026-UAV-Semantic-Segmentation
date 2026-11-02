from landcover.cli import command_arguments
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
from landcover.commands._sixview_predict_support import SixViewImages, stack_views, create_loader, close_loader, transfer_views, average_views, configure_attention, store_mask
from landcover.bootstrap import ROOT, prepare_runtime
from landcover.artifacts import file_digest, write_record
prepare_runtime()
PROJECT = ROOT

def execute():
    arguments, options = command_arguments('sixview_predict')
    destination = Path(options.output_root)
    destination.mkdir(parents=True, exist_ok=True)
    prediction_dir = destination / 'test2_predictions'
    prediction_dir.mkdir(exist_ok=True)
    if any(prediction_dir.iterdir()):
        raise RuntimeError('Use a fresh output directory; never mix checkpoints')
    image_paths = [Path(options.test_dir) / f'test2_{i}.png' for i in range(1, 1301)]
    if set(Path(options.test_dir).glob('*.png')) != set(image_paths):
        raise RuntimeError('Test2 expected exactly 1300 numbered images')
    shapes = set()
    for location in image_paths:
        with Image.open(location) as image:
            shapes.add(image.size)
    if len(shapes) != 1:
        raise RuntimeError(f'Mixed shapes need separate batches: {shapes}')
    record = dict(status='tuning', started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), checkpoint=options.original_checkpoint, checkpoint_sha256=file_digest(options.original_checkpoint), inference_checkpoint=options.checkpoint, inference_checkpoint_sha256=file_digest(options.checkpoint), config=options.config, config_sha256=file_digest(options.config), image_count=1300, views=6, scales=[0.75, 1.0, 1.125], flips=['identity', 'horizontal'], crop=[768, 768], stride=[512, 512], hann_minimum=0.125, barren_multiplier=1.0, dtype='float32', tf32=False, worker_benchmarks=[], batch_benchmarks=[])
    report_path = destination / 'inference_report.json'
    write_record(report_path, record)
    for worker_count in (4, 8, 12):
        batches = iter(create_loader(image_paths[:48], options.config, worker_count))
        next(batches)
        start_time = time.perf_counter()
        quantity = 0
        for loader_part in batches:
            quantity += len(loader_part['paths'])
        seconds = time.perf_counter() - start_time
        close_loader(batches)
        benchmark = dict(workers=worker_count, images=quantity, seconds=seconds, images_per_second=quantity / seconds)
        record['worker_benchmarks'].append(benchmark)
        print('WORKER_BENCHMARK', json.dumps(benchmark), flush=True)
        write_record(report_path, record)
        del batches, loader_part
        gc.collect()
    worker_count = max(record['worker_benchmarks'], key=lambda x: x['images_per_second'])['workers']
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from landcover.checkpoints import allow_project_weights
    allow_project_weights()
    import mmcv_custom
    import mmseg_custom
    from mmseg.apis import init_segmentor
    model = init_segmentor(options.config, options.checkpoint, device='cuda:0')
    model.eval()
    model.requires_grad_(False)
    model.barren_multiplier = 1.0
    dataset = SixViewImages(image_paths, options.config)
    sample_parts = [stack_views([dataset[i]]) for i in (0, 649, 1299)]
    comparisons = []
    for loader_part in sample_parts:
        view_batch = transfer_views([loader_part])
        reference = average_views(model, view_batch)
        attention_count = configure_attention(model, True)
        optimized = average_views(model, view_batch)
        max_diff = (reference - optimized).abs().max().item()
        disagreements = (reference.argmax(1) != optimized.argmax(1)).sum().item()
        comparisons.append(dict(max_probability_difference=max_diff, disagreeing_pixels=disagreements, pixels=reference.shape[0] * reference.shape[2] * reference.shape[3]))
        configure_attention(model, False)
        del view_batch, reference, optimized
    use_optimized = all((item['max_probability_difference'] <= 0.0001 and item['disagreeing_pixels'] == 0 for item in comparisons))
    configure_attention(model, use_optimized)
    record['attention_verification'] = dict(enabled=use_optimized, modules=attention_count, comparisons=comparisons)
    print('ATTENTION_VERIFICATION', json.dumps(record['attention_verification']), flush=True)
    write_record(report_path, record)
    for batch_size in (4, 8, 16, 24, 32, 40, 48, 56, 64):
        view_batch = None
        try:
            torch.cuda.empty_cache()
            loader_parts = [stack_views([dataset[i] for i in range(start, min(start + 4, batch_size))]) for start in range(0, batch_size, 4)]
            view_batch = transfer_views(loader_parts)
            del loader_parts
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_time = time.perf_counter()
            with torch.inference_mode():
                prediction = model.encode_decode(view_batch['img'][0], view_batch['metas'][0])
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start_time
            benchmark = dict(batch=batch_size, seconds=seconds, images_per_second=batch_size / seconds, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30)
            record['batch_benchmarks'].append(benchmark)
            print('BATCH_BENCHMARK', json.dumps(benchmark), flush=True)
            del prediction
            if benchmark['peak_allocated_gib'] > 71:
                break
        except torch.OutOfMemoryError:
            print(f'BATCH_OOM batch={batch_size}; earlier successful candidates retained', flush=True)
            record['batch_benchmarks'].append(dict(batch=batch_size, oom=True))
            break
        finally:
            del view_batch
            gc.collect()
            torch.cuda.empty_cache()
            write_record(report_path, record)
    choices = [item for item in record['batch_benchmarks'] if not item.get('oom') and item['peak_allocated_gib'] < 72]
    if not choices:
        raise RuntimeError('No safe batch size')
    fastest = max(choices, key=lambda x: x['images_per_second'])
    chosen = max((item for item in choices if item['images_per_second'] >= fastest['images_per_second'] * 0.98), key=lambda x: x['batch'])
    batch_size = chosen['batch']
    effective_size = batch_size
    record.update(status='predicting', batch_size=batch_size, workers=worker_count, selected_benchmark=chosen)
    write_record(report_path, record)
    print('SELECTED', json.dumps(dict(batch=batch_size, workers=worker_count, benchmark=chosen)), flush=True)
    del dataset, sample_parts
    gc.collect()
    batches = iter(create_loader(image_paths, options.config, worker_count))
    start_time = time.perf_counter()
    done = 0
    writes = []
    torch.cuda.reset_peak_memory_stats()
    with ThreadPoolExecutor(max_workers=4) as writer:
        while done < len(image_paths):
            loader_parts = []
            for _ in range(batch_size // 4):
                try:
                    loader_parts.append(next(batches))
                except StopIteration:
                    break
            view_batch = transfer_views(loader_parts)
            del loader_parts
            while True:
                try:
                    mask_chunks = []
                    for offset in range(0, len(view_batch['paths']), effective_size):
                        sub = dict(img=[view[offset:offset + effective_size] for view in view_batch['img']], metas=[view[offset:offset + effective_size] for view in view_batch['metas']])
                        mask_chunks.append(average_views(model, sub).argmax(1).cpu().numpy())
                    label_maps = np.concatenate(mask_chunks)
                    break
                except torch.OutOfMemoryError:
                    if effective_size <= 4:
                        raise
                    effective_size = max(4, effective_size // 2 // 4 * 4)
                    mask_chunks = []
                    gc.collect()
                    torch.cuda.empty_cache()
                    record.setdefault('runtime_memory_fallbacks', []).append(dict(at_image=done, batch=effective_size))
                    print(f'RUNTIME_MEMORY_FALLBACK batch={effective_size}', flush=True)
            for location, mask in zip(view_batch['paths'], label_maps):
                writes.append(writer.submit(store_mask, location, mask, prediction_dir))
            done += len(view_batch['paths'])
            del view_batch, label_maps
            for task in writes:
                task.result()
            writes.clear()
            seconds = time.perf_counter() - start_time
            record.update(completed_images=done, prediction_seconds=seconds, effective_compute_batch=effective_size, eta_minutes=(len(image_paths) - done) * seconds / max(done, 1) / 60, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30)
            write_record(report_path, record)
            print(f"PROGRESS {done}/1300 batch={batch_size} workers={worker_count} elapsed={seconds / 60:.2f}min eta={record['eta_minutes']:.2f}min peak={record['peak_reserved_gib']:.2f}GiB", flush=True)
    close_loader(batches)
    record.update(status='predictions_complete', completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    write_record(report_path, record)
if __name__ == '__main__':
    execute()
