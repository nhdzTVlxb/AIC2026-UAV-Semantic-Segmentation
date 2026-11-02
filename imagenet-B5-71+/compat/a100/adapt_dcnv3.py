"""Apply an idempotent PyTorch-2/CUDA-13 compatibility patch to a DCNv3 copy."""
from __future__ import annotations
import argparse
from pathlib import Path

def substitute_token(path: Path, old: str, new: str, minimum: int=0) -> int:
    text = path.read_text(encoding='utf-8')
    quantity = text.count(old)
    if quantity:
        path.write_text(text.replace(old, new), encoding='utf-8', newline='\n')
    elif minimum and new not in text:
        raise RuntimeError(f'expected token not found in {path}: {old!r}')
    return quantity

def execute() -> None:
    arguments = argparse.ArgumentParser()
    arguments.add_argument('root', type=Path, help='copied ops_dcnv3 directory')
    options = arguments.parse_args()
    destination = options.root.resolve()
    required = [destination / 'src' / 'dcnv3.h', destination / 'src' / 'cuda' / 'dcnv3_cuda.cu', destination / 'setup.py']
    for location in required:
        if not location.is_file():
            raise FileNotFoundError(location)
    header = destination / 'src' / 'dcnv3.h'
    cuda = destination / 'src' / 'cuda' / 'dcnv3_cuda.cu'
    im2col = destination / 'src' / 'cuda' / 'dcnv3_im2col_cuda.cuh'
    setup = destination / 'setup.py'
    changed = 0
    for location in (header, cuda):
        changed += substitute_token(location, '.type().is_cuda()', '.is_cuda()')
    changed += substitute_token(cuda, 'AT_ASSERTM', 'TORCH_CHECK')
    changed += substitute_token(cuda, 'input.type(),', 'input.scalar_type(),')
    changed += substitute_token(cuda, '.data<', '.data_ptr<')
    changed += substitute_token(im2col, '#include <THC/THCAtomics.cuh>\n', '')
    changed += substitute_token(setup, "extra_compile_args = {'cxx': []}", "extra_compile_args = {'cxx': ['-O3', '-std=c++17']}")
    changed += substitute_token(
        setup,
        'if torch.cuda.is_available() and CUDA_HOME is not None:',
        'if (torch.cuda.is_available() or os.getenv("AIC_FORCE_CUDA_BUILD") == "1") and CUDA_HOME is not None:',
        minimum=1,
    )
    changed += substitute_token(setup, "extra_compile_args['nvcc'] = [\n", "extra_compile_args['nvcc'] = ['-O3', '-std=c++17',\n")
    combined = '\n'.join((location.read_text(encoding='utf-8')
                           for location in (header, cuda, im2col, setup)))
    forbidden = ['.type().is_cuda()', '.data<', 'THC/THCAtomics.cuh',
                 'if torch.cuda.is_available() and CUDA_HOME is not None:']
    remaining = [token for token in forbidden if token in combined]
    if remaining:
        raise RuntimeError(f'legacy DCNv3 tokens remain: {remaining}')
    stamp = destination / '.aic_a100_torch2_patch'
    stamp.write_text('DCNv3 PyTorch-2/CUDA compatibility patch v1\n', encoding='utf-8')
    print(f'DCNv3 compatibility patch ready; replacements={changed}; root={destination}')
if __name__ == '__main__':
    execute()
