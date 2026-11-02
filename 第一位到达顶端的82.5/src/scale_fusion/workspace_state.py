from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TEAM = Path(os.environ.get('AIC3_OUTPUT_OWNER', ROOT / 'runs/aic3')).resolve()
if TEAM.name != 'aic3':
    raise ValueError('Training output must belong to the aic3 owner directory')
TRAIN_DATA = Path(os.environ.get('AIC3_PAIR_DIRECTORY', os.environ.get('AIC_DATA_ROOT', '/data/uav'))).resolve()
CONFIG_PATH = ROOT / 'config/逐像素尺度门.json'
PUBLIC_SHA = 'f3ce777e1f1192bc9f20ee5a9720bfd95edab60fa80462c83cc76c94c1ad8fce'
PUBLIC_FILE = Path(os.environ.get('AIC3_PUBLIC_FILE', Path('/models/pretrained/mtp_loveda_internimage_xl_f3ce777e.pth'))).resolve()
PUBLIC_SOURCES = ROOT / 'third_party'

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write_json(path, data):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(f'{destination.name}.{os.getpid()}.writing')
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    pending.replace(destination)

def local_path(relative):
    candidate = (TEAM / relative).resolve()
    if not candidate.is_relative_to(TEAM):
        raise ValueError('Training artifacts must remain in their owner directory')
    return candidate

def source_inventory():
    paths = sorted((ROOT / 'scale_fusion').rglob('*.py'))
    paths.extend([CONFIG_PATH, ROOT / 'config/训练两阶段.json', ROOT / 'train.py',
                  PUBLIC_SOURCES / 'provenance.json'])
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}

def freeze_sources(folder):
    sources = source_inventory()
    archive = Path(folder) / 'sources.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as output:
        for relative in sources:
            output.write(ROOT / relative, relative)
    return dict(source_files=sources, source_snapshot=str(archive.relative_to(TEAM)),
                source_snapshot_sha256=digest(archive))

def check_sources(record):
    if record['source_files'] != source_inventory():
        raise ValueError('The fixed scale-gate training sources changed during execution')

@contextmanager
def lease(gpus):
    if not gpus or len(gpus) != len(set(gpus)) or min(gpus) < 0:
        raise ValueError('Provide distinct nonnegative physical device indices')
    with ExitStack() as stack:
        for gpu in sorted(gpus):
            handle = stack.enter_context(open(f'/tmp/aic-reference-physical-gpu{gpu}.lock', 'a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            usage = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
                '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
            if int(usage.strip()) >= 512:
                raise RuntimeError(f'Physical GPU{gpu} is occupied ({usage.strip()} MiB)')
        yield

def load_own(path, checkpointing=True):
    import torch
    from scale_fusion.pyramid.segmenting_graph import ScaleAdaptiveSegmenter
    checkpoint = local_path(path)
    record = json.loads((checkpoint.parent / 'training.json').read_text())
    if record['status'] != 'trained' or record['owner_team'] != 'aic3':
        raise ValueError('A completed aic3 phase is required')
    if digest(checkpoint) != record['checkpoint_sha256']:
        raise ValueError('The own preceding checkpoint differs')
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    if payload['owner_team'] != 'aic3' or payload['lineage_steps'] != record['lineage_steps']:
        raise ValueError('The model owner or accumulated optimization count differs')
    model = ScaleAdaptiveSegmenter(checkpointing)
    model.load_state_dict(payload['model'], strict=True)
    return model, record
