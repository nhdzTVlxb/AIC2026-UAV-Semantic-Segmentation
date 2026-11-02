import numpy as np
import torch
from PIL import Image, ImageEnhance
from scale_fusion.workspace_state import TRAIN_DATA

def pairs():
    images = sorted((TRAIN_DATA / 'train/images').glob('*.png'))
    masks = sorted((TRAIN_DATA / 'train/masks').glob('*.png'))
    if len(images) != 6996 or [p.name for p in images] != [p.name for p in masks]:
        raise ValueError('Exactly 6996 official pairs required')
    return list(zip(images, masks))

def plan(count, seed):
    rng = np.random.default_rng(seed)
    indices = np.concatenate([rng.permutation(6996) for _ in range((count + 6995) // 6996)])[:count]
    return (indices, rng.integers(0, 2 ** 32, count, dtype=np.uint32))

class OfficialTiles(torch.utils.data.Dataset):

    def __init__(self, crop, indices, seeds):
        self.files = pairs()
        self.crop = crop
        self.indices = indices
        self.seeds = seeds

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image_path, mask_path = self.files[int(self.indices[index])]
        rng = np.random.default_rng(int(self.seeds[index]))
        with Image.open(image_path) as im:
            rgb = im.convert('RGB')
        with Image.open(mask_path) as im:
            mask = im.copy()
        if rgb.size != (1024, 1024) or mask.size != rgb.size or mask.mode != 'L':
            raise ValueError('Unexpected official input format')
        side = max(self.crop, round(1024 * rng.uniform(0.75, 1.3)))
        rgb = rgb.resize((side, side), Image.Resampling.BILINEAR)
        mask = mask.resize((side, side), Image.Resampling.NEAREST)
        x, y = map(int, rng.integers(0, side - self.crop + 1, 2))
        box = (x, y, x + self.crop, y + self.crop)
        rgb = rgb.crop(box)
        mask = mask.crop(box)
        orientation = int(rng.integers(8))
        if orientation >= 4:
            rgb = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        for _ in range(orientation % 4):
            rgb = rgb.transpose(Image.Transpose.ROTATE_90)
            mask = mask.transpose(Image.Transpose.ROTATE_90)
        for transform in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            rgb = transform(rgb).enhance(float(rng.uniform(0.86, 1.14)))
        image = torch.from_numpy(np.array(rgb, dtype=np.uint8)).permute(2, 0, 1)
        label = np.array(mask, dtype=np.int64) - 1
        label[label < 0] = 255
        if not np.isin(label, [*range(8), 255]).all():
            raise ValueError('Unexpected training label')
        return (image, torch.from_numpy(label))
