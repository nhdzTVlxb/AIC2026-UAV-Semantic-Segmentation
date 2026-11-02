import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from scale_fusion.workspace_state import CONFIG_PATH, ROOT, TEAM, TRAIN_DATA

def inspect_pairs(directory):
    from PIL import Image
    images = sorted((directory / 'train/images').glob('*.png'))
    labels = sorted((directory / 'train/masks').glob('*.png'))
    if len(images) != 6996 or [item.name for item in images] != [item.name for item in labels]:
        raise ValueError('6996 matching official training image/mask pairs are required')
    for position in (0, len(images) // 2, len(images) - 1):
        with Image.open(images[position]) as rgb, Image.open(labels[position]) as mask:
            if rgb.size != (1024, 1024) or mask.size != rgb.size or mask.mode != 'L':
                raise ValueError('The training pair shape or label mode differs')
    return dict(official_pairs=len(images), RGB_format_samples=3, optimizer_steps=0)

def run_course(devices, prepare=False):
    inventory = inspect_pairs(TRAIN_DATA)
    if prepare:
        print(json.dumps(inventory, ensure_ascii=False))
        return
    recipe = json.loads(CONFIG_PATH.read_text())
    order = json.loads((ROOT / 'config/训练两阶段.json').read_text())['stages']
    if order != ['vehicle_alignment', 'pyramid_joint'] or [p['id'] for p in recipe['stages']] != order:
        raise ValueError('The two-phase course changed')
    subprocess.run([sys.executable, '-m', 'scale_fusion.priors.checkpoint_fetch'], cwd=ROOT, check=True)
    parent = None
    for phase in order:
        command = [sys.executable, '-m', 'scale_fusion.fitting.device_dispatch',
                   '--gpus', *map(str, devices), '--stage', phase, '--tag', phase]
        if parent is not None:
            command.extend(['--parent', parent])
        subprocess.run(command, cwd=ROOT, check=True)
        parent = f'runs/{phase}/weights.pth'
    subprocess.run([sys.executable, '-m', 'scale_fusion.handover.trained_identity',
                    '--training-root', str(TEAM)], cwd=ROOT, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs='+', default=[1])
    parser.add_argument('--prepare-only', action='store_true')
    arguments = parser.parse_args()
    run_course(arguments.gpus, arguments.prepare_only)

if __name__ == '__main__':
    main()
