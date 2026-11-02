from __future__ import annotations
from landcover.cli import command_arguments
import argparse
import json
import os
from pathlib import Path
from landcover.bootstrap import ROOT, prepare_runtime
from landcover.artifacts import file_digest, write_record
prepare_runtime()

def execute() -> None:
    arguments, options = command_arguments('partition_holdout')
    ids = [line.strip() for line in options.val_split.read_text().splitlines() if line.strip()]
    if len(ids) < 20 or len(set(ids)) != len(ids):
        raise ValueError(f'unexpected validation split size/content: {len(ids)}')
    calibration_ids = ids[::2]
    gate_ids = ids[1::2]
    options.output_root.mkdir(parents=True, exist_ok=True)
    image_aliases = options.output_root / 'images'
    image_aliases.mkdir(parents=True, exist_ok=True)
    calibration_path = options.output_root / 'calibration_ids.txt'
    gate_path = options.output_root / 'gate_split.txt'
    calibration_path.write_text('\n'.join(calibration_ids) + '\n', encoding='utf-8')
    gate_path.write_text('\n'.join(gate_ids) + '\n', encoding='utf-8')
    for sample_id in ids:
        for folder in ('images', 'masks'):
            incoming = options.data_root / folder / f'{sample_id}.png'
            if not incoming.is_file():
                raise FileNotFoundError(incoming)
    for index, sample_id in enumerate(calibration_ids, 1):
        incoming = options.data_root / 'images' / f'{sample_id}.png'
        alias = image_aliases / f'test2_{index}.png'
        if alias.is_symlink():
            if alias.resolve() != incoming.resolve():
                raise RuntimeError(f'unexpected existing alias target: {alias}')
        elif alias.exists():
            raise RuntimeError(f'refusing to replace non-symlink alias: {alias}')
        else:
            os.symlink(incoming, alias)
    manifest = {'validation_count': len(ids), 'calibration_count': len(calibration_ids), 'gate_count': len(gate_ids), 'calibration_ids': str(calibration_path), 'gate_split': str(gate_path), 'image_alias_dir': str(image_aliases)}
    (options.output_root / 'split_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))
if __name__ == '__main__':
    execute()
