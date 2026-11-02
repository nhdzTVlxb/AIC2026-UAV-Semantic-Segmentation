import numpy as np
from PIL import Image

class SixViewPredictor:
    def __init__(self, graph, torch):
        self.graph = graph
        self.torch = torch
        self.window = 704
        self.scales = (.75, 1., 1.25)
        self.mean = torch.tensor([123.675,116.28,103.53], device='cuda')[None,:,None,None]
        self.std = torch.tensor([58.395,57.12,57.375], device='cuda')[None,:,None,None]
        axis = torch.linspace(-1.,1.,self.window,device='cuda')
        one = torch.exp(-.5*(axis/.38).square()).clamp_min(.10)
        self.mass = (one[:,None]*one[None,:]).reshape(1,1,self.window,self.window)

    def one_scale(self, images, scale):
        t = self.torch
        height,width = images[0].shape[:2]
        size = round(height*scale)
        if height != width or size < self.window or size > self.window*2:
            raise ValueError('Unsupported shape for the fixed submission protocol')
        corners = [(r,c) for r in (0,size-self.window) for c in (0,size-self.window)]
        groups=[]
        for rgb in images:
            scaled=np.asarray(Image.fromarray(rgb,mode='RGB').resize((size,size),Image.Resampling.BILINEAR),dtype=np.uint8)
            x=t.from_numpy(scaled.copy()).permute(2,0,1)[None].cuda().float()
            x=(x-self.mean)/self.std
            groups.append(t.cat([x[:,:,r:r+self.window,c:c+self.window] for r,c in corners],0))
        crops=t.cat(groups,0)
        views=t.cat([crops,crops.flip(-1)],0)
        with t.autocast('cuda',dtype=t.bfloat16):
            logits,auxiliary=self.graph(views)
        probabilities=logits.float().softmax(1)
        count=len(images)*len(corners)
        direct,mirror=probabilities.split(count,0)
        paired=(direct+mirror.flip(-1))*.5
        canvas=t.zeros((len(images),8,size,size),device='cuda')
        mass=t.zeros((1,1,size,size),device='cuda')
        for ordinal,(r,c) in enumerate(corners):
            canvas[:,:,r:r+self.window,c:c+self.window].add_(self.mass*paired[ordinal::len(corners)])
            mass[:,:,r:r+self.window,c:c+self.window].add_(self.mass)
        if not bool((mass>0).all()):
            raise RuntimeError('Incomplete window coverage')
        current=t.nn.functional.interpolate(canvas/mass,(height,width),mode='bilinear',align_corners=False)
        if not bool(current.isfinite().all()):
            raise FloatingPointError('Non-finite probabilities')
        return current

    def probabilities(self,images):
        t=self.torch
        with t.inference_mode():
            result=None
            for scale in self.scales:
                p=self.one_scale(images,scale)
                result=p if result is None else result+p
            return result/len(self.scales)

    def labels(self,images):
        with self.torch.inference_mode():
            return self.probabilities(images).argmax(1).byte().cpu().numpy()+1
