"""Locate the checkout independently of the command's working directory."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION = ROOT / "vendor" / "InternImage" / "segmentation"


def prepare_runtime(trusted=False):
    for location in (ROOT, SEGMENTATION):
        value = str(location)
        if value not in sys.path:
            sys.path.insert(0, value)
    os.environ.setdefault("AIC_PROJECT_ROOT", str(ROOT))
    if trusted:
        from landcover.checkpoints import allow_project_weights
        allow_project_weights()
    return ROOT
