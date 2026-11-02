from pathlib import Path
import argparse
import os
import numpy as np
from PIL import Image
from common import atomic_json,configure,digest,load_model

def main():
    parser=argparse.ArgumentParser()
    data=Path(os.environ.get('AIC_DATA_ROOT','/data/uav'))
    parser.add_argument('--images',type=Path,default=data/'train/images')
    parser.add_argument('--masks',type=Path,default=data/'train/masks')
    parser.add_argument('--checkpoint',type=Path,default=Path(os.environ.get('AIC_CHECKPOINT','/models/checkpoints/best.pth')))
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--count',type=int,default=700)
    parser.add_argument('--seed',type=int,default=3407)
    parser.add_argument('--report',type=Path,default=Path('/outputs/validation.json'))
    args=parser.parse_args()
    configure()
    from scale_fusion.workspace_state import lease
    from inference import SixViewPredictor
    files=sorted(args.images.glob('*.png'))
    if not files:raise ValueError('No validation images')
    if args.count>0 and args.count<len(files):
        indices=np.random.default_rng(args.seed).choice(len(files),args.count,replace=False)
        files=[files[i] for i in sorted(indices)]
    cm=np.zeros((8,8),dtype=np.int64)
    with lease([args.gpu]):
        t,model=load_model(args.checkpoint,args.gpu)
        painter=SixViewPredictor(model,t)
        for done,path in enumerate(files,1):
            with Image.open(path) as im:rgb=np.asarray(im.convert('RGB'))
            with Image.open(args.masks/path.name) as im:truth=np.asarray(im,dtype=np.int64)
            pred=painter.labels([rgb])[0].astype(np.int64)
            valid=(truth>=1)&(truth<=8)
            cm+=np.bincount((truth[valid]-1)*8+pred[valid]-1,minlength=64).reshape(8,8)
            if done%25==0 or done==len(files):print('VALIDATE',done,'/',len(files),flush=True)
    tp=np.diag(cm).astype(float)
    den=cm.sum(0)+cm.sum(1)-tp
    iou=np.divide(tp,den,out=np.zeros(8),where=den>0)
    atomic_json(args.report,{'count':len(files),'miou_pct':float(iou.mean()*100),'iou_pct':(iou*100).tolist(),'confusion':cm.tolist(),'checkpoint_sha256':digest(args.checkpoint),'images_directory':str(args.images),'masks_directory':str(args.masks),'evaluation_scope':'training_set_diagnostic' if args.images.resolve()==(data/'train/images').resolve() else 'provided_validation_set','files':[p.name for p in files]})
    print('MIOU_PCT',float(iou.mean()*100),flush=True)

if __name__=='__main__':
    main()
