from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from landcover.bootstrap import ROOT, prepare_runtime
from landcover.artifacts import file_digest, write_record
prepare_runtime()
CLASS_NAMES = ['Background', 'Building', 'Road', 'Water', 'Barren', 'Vegetation', 'Agricultural', 'Vehicle']

def load_q8(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        q8 = payload['probs_q8']
    if q8.dtype != np.uint8 or q8.shape != (8, 1024, 1024):
        raise ValueError(f'invalid q8 payload at {path}: {q8.dtype} {q8.shape}')
    return q8
