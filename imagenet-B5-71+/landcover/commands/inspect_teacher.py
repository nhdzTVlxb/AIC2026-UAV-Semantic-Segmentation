from __future__ import annotations
from landcover.cli import command_arguments
import argparse
import json
from pathlib import Path
import numpy as np
from landcover.bootstrap import ROOT, prepare_runtime
from landcover.artifacts import file_digest, write_record
prepare_runtime()

def execute() -> None:
    arguments, options = command_arguments('inspect_teacher')
    record = json.loads(options.report.read_text(encoding='utf-8'))
    files = sorted(options.prob_dir.glob('test2_*.npz'))
    if len(files) != options.expected_count:
        raise RuntimeError(f'probability count {len(files)} != {options.expected_count}')
    if record.get('image_count') != options.expected_count:
        raise RuntimeError('teacher report image_count mismatch')
    if record.get('checkpoint_sha256') != options.checkpoint_sha256:
        raise RuntimeError('teacher checkpoint hash mismatch')
    hard = float(record.get('fraction_ge_090', -1))
    soft_region = float(record.get('fraction_050_to_090', -1))
    if not 0.03 <= hard <= 0.65:
        raise RuntimeError(f'implausible hard-pseudo fraction: {hard}')
    if not 0.25 <= soft_region <= 0.97:
        raise RuntimeError(f'implausible rejected-KL fraction: {soft_region}')
    for index in (0, len(files) // 4, len(files) // 2, 3 * len(files) // 4, len(files) - 1):
        with np.load(files[index]) as payload:
            probability = payload['probs_q8']
        if probability.shape != (8, 1024, 1024) or probability.dtype != np.uint8:
            raise RuntimeError(f'invalid probability file: {files[index]}')
        if np.any(probability.sum(axis=0) == 0):
            raise RuntimeError(f'zero-sum probability pixel: {files[index]}')
    print('PSEUDO_TEACHER_VALID')
    print('count', len(files))
    print('hard_fraction', hard)
    print('rejected_fraction', soft_region)
    print('teacher_seconds', record.get('total_seconds'))
if __name__ == '__main__':
    execute()
