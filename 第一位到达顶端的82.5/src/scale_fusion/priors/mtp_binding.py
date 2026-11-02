from __future__ import annotations
import ast
import functools
import json
import sys
import types
import zipfile
import torch
from torch.nn import functional as F
from scale_fusion.workspace_state import PUBLIC_SOURCES, PUBLIC_FILE, PUBLIC_SHA, digest

@functools.lru_cache(maxsize=1)
def public_archives():
    directory = PUBLIC_SOURCES
    provenance = json.loads((directory / 'provenance.json').read_text())
    for item in ('public_Python_libraries', 'public_model_sources'):
        entry = provenance[item]
        if digest(directory / entry['archive']) != entry['sha256']:
            raise ValueError('Public dependency SHA mismatch: ' + item)
    sys.path.insert(0, str(directory / provenance['public_Python_libraries']['archive']))
    return (directory / provenance['public_model_sources']['archive'], provenance)

def read_source(filename):
    import hashlib
    archive, provenance = public_archives()
    with zipfile.ZipFile(archive) as package:
        raw = package.read(filename)
    expected = provenance['public_model_sources']['entry_sha256'][filename]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Upstream mathematical source SHA mismatch')
    return (raw.decode(), f'{archive}/{filename}')

@functools.lru_cache(maxsize=1)
def encoder_class():
    functions, origin = read_source('dcnv3_func.py')
    tree = ast.parse(functions)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {'torch': torch, 'F': F}
    exec(compile(tree, origin, 'exec'), namespace)
    operators, origin = read_source('dcnv3.py')
    tree = ast.parse(operators)
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.level or (isinstance(node, ast.ClassDef) and node.name == 'DCNv3'))]
    modules = {'dcnv3_core_pytorch': namespace['dcnv3_core_pytorch']}
    exec(compile(tree, origin, 'exec'), modules)
    reference = modules['DCNv3_pytorch']

    class FullPrecisionDeformable(reference):

        def forward(self, features):
            with torch.autocast(features.device.type, enabled=False):
                value = super().forward(features.float())
            return value.to(features.dtype)
    backbone, origin = read_source('intern_image.py')
    tree = ast.parse(backbone)
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and (node.module == 'mmseg.registry' or node.level))]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'InternImage':
            node.decorator_list = []
    module = types.ModuleType('_aic3_scale_gate_internimage')
    module.__file__ = origin
    module.opsm = types.SimpleNamespace(DCNv3_pytorch=FullPrecisionDeformable)
    sys.modules[module.__name__] = module
    exec(compile(tree, origin, 'exec'), module.__dict__)
    return module.InternImage

def load_public_parameters():
    import numpy as np
    public_archives()
    from mmengine.logging.history_buffer import HistoryBuffer

    def numeric_log_attribute(owner, name):
        if owner is not HistoryBuffer or name not in {'min', 'max', 'current', 'mean'}:
            raise ValueError('Unexpected pickled log attribute')
        return getattr(owner, name)
    supported = {'builtins.getattr': numeric_log_attribute, 'mmengine.logging.history_buffer.HistoryBuffer': HistoryBuffer, 'numpy.ndarray': np.ndarray, 'numpy.dtype': np.dtype}
    for prefix in ('numpy.core.multiarray', 'numpy._core.multiarray'):
        supported[prefix + '._reconstruct'] = np._core.multiarray._reconstruct
        supported[prefix + '.scalar'] = np._core.multiarray.scalar
    if digest(PUBLIC_FILE) != PUBLIC_SHA:
        raise ValueError('The public LoveDA checkpoint changed')
    if hasattr(torch.serialization, 'get_unsafe_globals_in_checkpoint'):
        requested = torch.serialization.get_unsafe_globals_in_checkpoint(PUBLIC_FILE)
        if set(requested) - set(supported):
            raise ValueError('Unexpected pickle globals: ' + repr(requested))
        permitted = [(supported[name], name) for name in requested]
        permitted += [np.dtypes.Float64DType, np.dtypes.Float32DType, np.dtypes.Int64DType, np.dtypes.Int32DType, np.dtypes.UInt32DType]
        with torch.serialization.safe_globals(permitted):
            saved = torch.load(PUBLIC_FILE, map_location='cpu', weights_only=True, mmap=True)
    else:

        saved = torch.load(PUBLIC_FILE, map_location='cpu', weights_only=False, mmap=True)
    state = saved['state_dict']
    if len(state) != 949 or state['decode_head.conv_seg.weight'].shape != (7, 512, 1, 1):
        raise ValueError('Expected the seven-class complete MTP task model')
    return state
