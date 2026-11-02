import argparse
import os
import subprocess
import sys
from scale_fusion.workspace_state import ROOT, lease

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs='+', default=[1])
    parser.add_argument('--stage', choices=['vehicle_alignment', 'pyramid_joint'], required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--parent')
    parser.add_argument('--probe', action='store_true')
    job = parser.parse_args()
    if len(job.gpus) not in (1, 2):
        raise ValueError('Use one accumulated device or two synchronized devices')
    environment = os.environ | dict(CUDA_VISIBLE_DEVICES=','.join(map(str, job.gpus)),
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', NCCL_P2P_DISABLE='1')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
        f'--nproc_per_node={len(job.gpus)}', '--module', 'scale_fusion.fitting.partition_update',
        '--stage', job.stage, '--tag', job.tag, '--physical-gpus', *map(str, job.gpus)]
    if job.parent is not None:
        command.extend(['--parent', job.parent])
    if job.probe:
        command.append('--probe')
    with lease(job.gpus):
        subprocess.run(command, cwd=ROOT, env=environment, check=True)

if __name__ == '__main__':
    main()
