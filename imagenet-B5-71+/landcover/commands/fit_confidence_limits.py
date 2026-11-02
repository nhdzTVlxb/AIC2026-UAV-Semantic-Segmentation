from __future__ import annotations
from landcover.cli import command_arguments
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from landcover.commands._fit_confidence_limits_support import load_q8
from landcover.bootstrap import ROOT, prepare_runtime
from landcover.artifacts import file_digest, write_record
prepare_runtime()
CLASS_NAMES = ['Background', 'Building', 'Road', 'Water', 'Barren', 'Vegetation', 'Agricultural', 'Vehicle']

def execute() -> None:
    arguments, options = command_arguments('fit_confidence_limits')
    ids = [line.strip() for line in options.ids_file.read_text().splitlines() if line.strip()]
    if not ids:
        raise ValueError('empty calibration ID list')
    counts = np.zeros((8, 256), dtype=np.int64)
    correct = np.zeros_like(counts)
    for index, sample_id in enumerate(ids, 1):
        q8 = load_q8(options.prob_dir / f'test2_{index}.npz')
        mask_path = options.mask_dir / f'{sample_id}.png'
        raw = np.asarray(Image.open(mask_path), dtype=np.uint8)
        if raw.shape != (1024, 1024):
            raise ValueError(f'invalid mask shape {mask_path}: {raw.shape}')
        prediction = q8.argmax(axis=0)
        certainty = q8.max(axis=0)
        valid = (raw >= 1) & (raw <= 8)
        expected = raw.astype(np.int16) - 1
        flat_index = prediction[valid].astype(np.int64) * 256 + certainty[valid]
        counts += np.bincount(flat_index, minlength=8 * 256).reshape(8, 256)
        is_correct = valid & (prediction == expected)
        correct_index = prediction[is_correct].astype(np.int64) * 256 + certainty[is_correct]
        correct += np.bincount(correct_index, minlength=8 * 256).reshape(8, 256)
    class_limits = []
    per_class = []
    start_bin = int(np.ceil(options.min_threshold * 255.0))
    for class_id, class_name in enumerate(CLASS_NAMES):
        chosen_bin = None
        chosen_count = 0
        chosen_precision = None
        for threshold_bin in range(start_bin, 256):
            accepted_count = int(counts[class_id, threshold_bin:].sum())
            if accepted_count < options.min_pixels:
                continue
            correct_count = int(correct[class_id, threshold_bin:].sum())
            precision = correct_count / accepted_count
            if precision >= options.target_precision:
                chosen_bin = threshold_bin
                chosen_count = accepted_count
                chosen_precision = precision
                break
        if chosen_bin is None:
            threshold = options.fallback_threshold
            accepted_count = int(counts[class_id, int(np.ceil(threshold * 255)):].sum())
            correct_count = int(correct[class_id, int(np.ceil(threshold * 255)):].sum())
            precision = correct_count / accepted_count if accepted_count else None
            incoming = 'fallback'
        else:
            threshold = chosen_bin / 255.0
            accepted_count = chosen_count
            precision = chosen_precision
            incoming = 'calibrated'
        class_limits.append(round(float(threshold), 6))
        per_class.append({'class_id': class_id, 'class': class_name, 'accept_threshold': round(float(threshold), 6), 'estimated_precision': None if precision is None else round(float(precision), 6), 'accepted_pixels': accepted_count, 'threshold_source': incoming})
    record = {'calibration_images': len(ids), 'target_precision': options.target_precision, 'minimum_accepted_pixels': options.min_pixels, 'thresholds': class_limits, 'classes': per_class}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
if __name__ == '__main__':
    execute()
