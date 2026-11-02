from pathlib import Path
import argparse
import importlib.metadata as metadata
import json
import os
from common import MODEL_SHA,PUBLIC_SHA,configure,digest

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args()
    configure()
    public=Path(os.environ['AIC3_PUBLIC_FILE'])
    checkpoint=Path(os.environ.get('AIC_CHECKPOINT','/models/checkpoints/best.pth'))
    if digest(public)!=PUBLIC_SHA or digest(checkpoint)!=MODEL_SHA:raise ValueError('Model checksum mismatch')
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import torch
    from scale_fusion.pyramid.segmenting_graph import ScaleAdaptiveSegmenter
    model=ScaleAdaptiveSegmenter(checkpointing=False)
    initialization=model.initialize_public()
    if not initialization['original_public_tensors_exact']:raise ValueError('Public initialization differs')
    print(json.dumps({'status':'passed','public_sha256':PUBLIC_SHA,'checkpoint_sha256':MODEL_SHA,'public_tensors':initialization['retained_tensors'],'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':metadata.version('numpy'),'pillow':metadata.version('Pillow'),'timm':metadata.version('timm')},indent=2),flush=True)

if __name__=='__main__':
    main()
