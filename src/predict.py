from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
import numpy as np
from PIL import Image
from common import MODEL_SHA,atomic_json,configure,digest,load_model

def worker(args):
    configure()
    from scale_fusion.workspace_state import lease
    from inference import SixViewPredictor
    files=sorted(args.source.glob('*.png'))
    if not files:raise ValueError('No input PNG images')
    if args.limit:files=files[:args.limit]
    mine=files[args.rank::args.world_size]
    labels=args.output/'labels'
    labels.mkdir(parents=True,exist_ok=True)
    with lease([args.worker]):
        t,graph=load_model(args.checkpoint,args.worker)
        painter=SixViewPredictor(graph,t)
        batch=args.image_batch
        index=0
        started=time.monotonic()
        while index<len(mine):
            group=mine[index:index+batch]
            images=[]
            for path in group:
                with Image.open(path) as im:rgb=np.asarray(im.convert('RGB'))
                if rgb.shape!=(1024,1024,3):raise ValueError(f'Invalid input {path}')
                images.append(rgb)
            try:
                predictions=painter.labels(images)
            except t.cuda.OutOfMemoryError:
                if batch==1:raise
                batch=max(1,batch//2)
                t.cuda.empty_cache()
                continue
            for path,label in zip(group,predictions):
                target=labels/path.name
                pending=target.with_name(target.name+f'.{os.getpid()}.tmp')
                Image.fromarray(label,mode='L').save(pending,format='PNG')
                pending.replace(target)
            index+=len(group)
            if index%25<len(group) or index==len(mine):
                progress={'rank':args.rank,'done':index,'total':len(mine),'batch':batch,'seconds':time.monotonic()-started}
                atomic_json(args.output/f'progress_{args.rank}.json',progress)
                print(json.dumps(progress),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,default=Path(os.environ.get('AIC_CHECKPOINT','/models/checkpoints/best.pth')))
    parser.add_argument('--source',type=Path,default=Path(os.environ.get('AIC_DATA_ROOT','/data/uav'))/'test_3/images')
    parser.add_argument('--output',type=Path,default=Path('/outputs/prediction'))
    parser.add_argument('--gpus',type=int,nargs='+',default=[0])
    parser.add_argument('--image-batch',type=int,default=2)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--worker',type=int)
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--world-size',type=int,default=1)
    args=parser.parse_args()
    if args.image_batch<1 or args.limit<0:parser.error('Invalid batch/limit')
    if args.worker is not None:
        worker(args);return
    if not args.gpus or len(args.gpus)!=len(set(args.gpus)):parser.error('Provide distinct GPU indices')
    configure()
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint_sha=digest(args.checkpoint)
    protocol={'checkpoint_sha256':checkpoint_sha,'scales':[.75,1.,1.25],'views':['identity','horizontal_flip'],'window':704,'fusion':'equal_probability_mean','class_multipliers':[1]*8,'limit':args.limit,'image_batch':args.image_batch}
    binding=args.output/'protocol.json'
    if binding.exists() and json.loads(binding.read_text())!=protocol:raise ValueError('Output belongs to another prediction protocol')
    atomic_json(binding,protocol)
    processes=[]
    for rank,gpu in enumerate(args.gpus):
        command=[sys.executable,str(Path(__file__).resolve()),'--checkpoint',str(args.checkpoint),'--source',str(args.source),'--output',str(args.output),'--image-batch',str(args.image_batch),'--limit',str(args.limit),'--worker',str(gpu),'--rank',str(rank),'--world-size',str(len(args.gpus))]
        processes.append(subprocess.Popen(command))
    codes=[p.wait() for p in processes]
    if any(codes):raise RuntimeError(f'Prediction workers failed: {codes}')
    source=sorted(args.source.glob('*.png'))
    if args.limit:source=source[:args.limit]
    expected={p.name for p in source}
    if not args.limit and expected!={f'test3_{i}.png' for i in range(1,3595)}:raise ValueError('Expected all3594 test3 image names')
    root=args.output/'labels'
    if {p.name for p in root.glob('*.png')}!=expected:raise ValueError('Prediction names/count differ')
    for p in root.glob('*.png'):
        with Image.open(p) as im:
            if im.mode!='L' or im.size!=(1024,1024):raise ValueError(f'Invalid output {p}')
            lo,hi=im.getextrema()
            if lo<1 or hi>8:raise ValueError(f'Invalid label range {p}')
    archive=args.output/'AIC-2026-64211460-test3.zip'
    pending=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(expected):z.write(root/name,name)
    pending.replace(archive)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC error')
    atomic_json(args.output/'prediction.json',{'count':len(expected),'zip':str(archive),'sha256':digest(archive),'checkpoint_sha256':checkpoint_sha})
    print('PREDICTION_COMPLETE',archive,flush=True)

if __name__=='__main__':
    main()
