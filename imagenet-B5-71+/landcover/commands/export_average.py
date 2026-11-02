from __future__ import annotations
from landcover.cli import command_arguments
import argparse
import hashlib
from pathlib import Path
import torch
from landcover.bootstrap import ROOT, prepare_runtime
from landcover.artifacts import file_digest, write_record
prepare_runtime()

def execute() -> None:
    arguments, options = command_arguments('export_average')
    if not 0.0 < options.ema_decay < 1.0:
        arguments.error('--ema-decay must be between 0 and 1')
    if options.output.exists():
        raise FileExistsError(f'refusing to overwrite: {options.output}')
    weight_record = torch.load(options.input, map_location='cpu', weights_only=False)
    if not isinstance(weight_record, dict) or 'state_dict' not in weight_record:
        raise RuntimeError('expected an MMCV checkpoint containing state_dict')
    tensor_state = weight_record['state_dict']
    ema_backup_keys = [key for key in tensor_state if key.startswith('ema_')]
    if not ema_backup_keys:
        raise RuntimeError('no EMA backup buffers found; checkpoint identity is unexpected')
    inference_state = {key: value for key, value in tensor_state.items() if not key.startswith('ema_')}
    weight_record['state_dict'] = inference_state
    weight_record.setdefault('meta', {})['aic_identity'] = f'ema_{options.ema_decay:g}_stripped_for_inference'
    weight_record['meta']['ema_decay'] = options.ema_decay
    weight_record['meta']['removed_ema_raw_backup_buffers'] = len(ema_backup_keys)
    torch.save(weight_record, options.output)
    print({'input': str(options.input), 'output': str(options.output), 'removed_buffers': len(ema_backup_keys), 'remaining_tensors': len(inference_state), 'size_bytes': options.output.stat().st_size, 'sha256': file_digest(options.output)})
if __name__ == '__main__':
    execute()
