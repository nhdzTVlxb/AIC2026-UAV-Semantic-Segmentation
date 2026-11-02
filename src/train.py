from pathlib import Path
import argparse
import os
import subprocess
import sys
from common import configure

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpus',nargs='+',type=int,default=[0,1])
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--backward-check',action='store_true')
    args=parser.parse_args()
    configure()
    root=Path(__file__).resolve().parent
    if args.backward_check:
        for stage in ['vehicle_alignment','pyramid_joint']:
            command=[sys.executable,'-m','scale_fusion.fitting.device_dispatch','--gpus',*map(str,args.gpus),'--stage',stage,'--tag',f'check_{stage}','--probe']
            subprocess.run(command,cwd=root,env=os.environ,check=True)
    else:
        command=[sys.executable,'-m','scale_fusion.fitting.two_phase_course','--gpus',*map(str,args.gpus)]
        if args.prepare_only:command.append('--prepare-only')
        subprocess.run(command,cwd=root,env=os.environ,check=True)

if __name__=='__main__':
    main()
