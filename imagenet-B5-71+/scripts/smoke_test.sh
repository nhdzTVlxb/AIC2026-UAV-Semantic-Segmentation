#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
export AIC_PROJECT_ROOT="$root"
export PYTHONPATH="$root:$root/vendor/InternImage/segmentation${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${PYTHON_BIN:-python}"
command -v "$python_bin" >/dev/null 2>&1 || python_bin=python3

"$python_bin" - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
recipes = json.loads((root / 'landcover' / 'recipes.json').read_text())['experiments']
required = [f'phase{i:02d}_' for i in range(1, 9)]
missing = [prefix for prefix in required if not any(name.startswith(prefix) for name in recipes)]
if missing:
    raise SystemExit(f'missing recipe keys: {missing}')
for number in range(1, 9):
    path = root / 'experiments' / f'phase{number:02d}_'
    matches = list(path.parent.glob(path.name + '*.py'))
    if not matches:
        raise SystemExit(f'missing stage recipe: {path.name}')
source = (root / 'run_predict.py').read_text()
expected = 'stage8_refresh_stage7_832_best_p1_test2.zip'
if expected not in source:
    raise SystemExit('prediction output filename is not pinned')
print('recipe_keys=8')
print('prediction_output=' + expected)
PY
"$python_bin" -m compileall -q run_train.py run_predict.py landcover experiments scripts compat vendor/InternImage/segmentation/mmseg_custom vendor/InternImage/segmentation/mmcv_custom
"$python_bin" - <<'PY'
from pathlib import Path
from mmcv import Config

configs = sorted(Path("experiments").glob("phase0[1-8]_*.py"))
if len(configs) != 8:
    raise SystemExit(f"expected 8 stage configurations, found {len(configs)}")
for path in configs:
    config = Config.fromfile(str(path.resolve()))
    if not config.get("model") or not config.get("data") or not config.get("work_dir"):
        raise SystemExit(f"incomplete stage configuration: {path}")
    print(f"{path.name}=CONFIG_OK")
PY
"$python_bin" - <<'PY'
import torch, torchvision, mmcv, mmseg, mmdet
import DCNv3
import mmcv_custom
import mmseg_custom
print('torch=' + torch.__version__)
print('torchvision=' + torchvision.__version__)
print('mmcv=' + mmcv.__version__)
print('mmseg=' + mmseg.__version__)
print('mmdet=' + mmdet.__version__)
print('cuda_available=' + str(torch.cuda.is_available()))
PY
"$python_bin" run_train.py --help >/dev/null
"$python_bin" run_predict.py --help >/dev/null
"$python_bin" -m landcover.commands.seal_submission --help >/dev/null
echo 'SMOKE_TEST_OK (no training or prediction run)'
