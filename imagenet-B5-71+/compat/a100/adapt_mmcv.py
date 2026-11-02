"""Apply PyTorch-2 CUDA/scatter compatibility patches to locked MMCV 1.7.2."""
from __future__ import annotations
import argparse
from pathlib import Path

def execute() -> None:
    arguments = argparse.ArgumentParser()
    arguments.add_argument('root', type=Path)
    options = arguments.parse_args()
    destination = options.root.resolve()
    helper = destination / 'mmcv' / 'ops' / 'csrc' / 'common' / 'pytorch_cuda_helper.hpp'
    deform = destination / 'mmcv' / 'ops' / 'csrc' / 'pytorch' / 'cuda' / 'ms_deform_attn_cuda.cu'
    kernel = destination / 'mmcv' / 'ops' / 'csrc' / 'common' / 'cuda' / 'ms_deform_attn_cuda_kernel.cuh'
    scatter = destination / 'mmcv' / 'parallel' / '_functions.py'
    for location in (helper, deform, kernel, scatter):
        if not location.is_file():
            raise FileNotFoundError(location)
    changed = 0
    for location in (helper, deform):
        text = location.read_text(encoding='utf-8')
        old = '#include <THC/THCAtomics.cuh>'
        if old in text:
            text = text.replace(old, '#include <ATen/cuda/Atomic.cuh>')
            location.write_text(text, encoding='utf-8', newline='\n')
            changed += 1
    text = kernel.read_text(encoding='utf-8')
    if 'atomicAdd(' in text:
        quantity = text.count('atomicAdd(')
        kernel.write_text(text.replace('atomicAdd(', 'gpuAtomicAdd('), encoding='utf-8', newline='\n')
        changed += quantity
    text = scatter.read_text(encoding='utf-8')
    old = 'streams = [_get_stream(device) for device in target_gpus]'
    new = "streams = [_get_stream(torch.device('cuda', device) if isinstance(device, int) else device) for device in target_gpus]"
    if old in text:
        scatter.write_text(text.replace(old, new), encoding='utf-8', newline='\n')
        changed += 1
    elif new not in text:
        raise RuntimeError(f'unexpected MMCV scatter implementation: {scatter}')
    leftovers = []
    for location in (helper, deform):
        if 'THC/THCAtomics.cuh' in location.read_text(encoding='utf-8'):
            leftovers.append(str(location))
    if leftovers:
        raise RuntimeError(f'legacy THC header remains: {leftovers}')
    (destination / '.aic_a100_torch2_patch').write_text('MMCV 1.7.2 PyTorch-2 CUDA/scatter compatibility patch v2\n', encoding='utf-8')
    print(f'MMCV compatibility patch ready; replacements={changed}; root={destination}')
if __name__ == '__main__':
    execute()
