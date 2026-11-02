from pathlib import Path
import hashlib
import os

MODEL_SHA = 'e52cb187727786e2fd8b2b90b50e2a6f874afaba651743b04625a7a382f1fd4d'
PUBLIC_SHA = 'f3ce777e1f1192bc9f20ee5a9720bfd95edab60fa80462c83cc76c94c1ad8fce'

def configure():
    os.environ.setdefault('AIC3_PAIR_DIRECTORY', os.environ.get('AIC_DATA_ROOT', '/data/uav'))
    os.environ.setdefault('AIC3_OUTPUT_OWNER', os.environ.get('AIC_OUTPUT', '/outputs/aic3'))
    os.environ.setdefault('AIC3_PUBLIC_FILE', os.environ.get('AIC_PRETRAINED', '/models/pretrained/mtp_loveda_internimage_xl_f3ce777e.pth'))
    os.environ.setdefault('AIC3_GRADIENT_CHECKPOINTING', '1')

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def load_model(checkpoint, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    configure()
    import torch
    from scale_fusion.pyramid.segmenting_graph import ScaleAdaptiveSegmenter
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    if saved.get('owner_team') != 'aic3' or saved.get('lineage_steps') != 1800:
        raise ValueError('Expected a completed two-stage checkpoint')
    graph = ScaleAdaptiveSegmenter(checkpointing=False)
    graph.load_state_dict(saved['model'], strict=True)
    del saved
    return torch, graph.cuda().eval()

def atomic_json(path, data):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+f'.{os.getpid()}.tmp')
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    pending.replace(path)
