from __future__ import annotations
from landcover.cli import command_arguments
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from landcover.bootstrap import ROOT, prepare_runtime
from landcover.artifacts import file_digest, write_record
prepare_runtime()

def execute() -> None:
    checkout = ROOT
    arguments, options = command_arguments('seal_submission')
    expected_names = [f'test2_{index}.png' for index in range(1, options.expected_count + 1)]
    actual_names = sorted((location.name for location in options.prediction_dir.glob('*.png')))
    if set(actual_names) != set(expected_names) or len(actual_names) != options.expected_count:
        missing = sorted(set(expected_names) - set(actual_names))[:20]
        extra = sorted(set(actual_names) - set(expected_names))[:20]
        raise RuntimeError(f'prediction set invalid: count={len(actual_names)}, missing={missing}, extra={extra}')
    entries = []
    aggregate_hist = np.zeros(9, dtype=np.int64)
    for name in expected_names:
        prediction_path = options.prediction_dir / name
        source_path = options.test_dir / name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        with Image.open(prediction_path) as image:
            prediction = np.asarray(image)
            mode = image.mode
        with Image.open(source_path) as incoming:
            expected_shape = (incoming.height, incoming.width)
        unique = np.unique(prediction)
        if prediction.ndim != 2 or prediction.shape != expected_shape:
            raise RuntimeError(f'invalid shape for {name}: {prediction.shape}, expected {expected_shape}')
        if unique.min() < 1 or unique.max() > 8:
            raise RuntimeError(f'invalid labels for {name}: {unique.tolist()}')
        if mode not in ('L', 'P'):
            raise RuntimeError(f'invalid PNG mode for {name}: {mode}')
        aggregate_hist += np.bincount(prediction.reshape(-1), minlength=9)[:9]
        entries.append({'name': name, 'sha256': file_digest(prediction_path), 'labels': unique.tolist()})
    options.output.parent.mkdir(parents=True, exist_ok=True)
    if options.output.exists():
        raise FileExistsError(f'refusing to overwrite existing ZIP: {options.output}')
    with zipfile.ZipFile(options.output, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name in expected_names:
            data = (options.prediction_dir / name).read_bytes()
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 24, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 33188 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(options.output, 'r') as archive:
        archived_names = archive.namelist()
        if archived_names != expected_names or archive.testzip() is not None:
            raise RuntimeError('post-write ZIP verification failed')
    manifest = {'zip': str(options.output.resolve()), 'zip_size_bytes': options.output.stat().st_size, 'zip_sha256': file_digest(options.output), 'file_count': len(entries), 'archive_layout': '1300 PNG files at ZIP root', 'label_range': [1, 8], 'aggregate_pixel_hist_0_to_8': aggregate_hist.tolist(), 'checkpoint': str(options.checkpoint.resolve()) if options.checkpoint else None, 'checkpoint_sha256': file_digest(options.checkpoint) if options.checkpoint else None, 'files': entries}
    manifest_path = options.output.with_suffix(options.output.suffix + '.manifest.json')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in manifest.items() if key != 'files'}, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    execute()
