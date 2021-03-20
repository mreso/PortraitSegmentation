import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class RandomNoise(object):
    def __init__(self, scale=10.0):
        self.scale = scale

    def __call__(self, x):
        if random.random() < 0.5:
            return x

        assert isinstance(x, torch.Tensor)

        std = self.scale * random.random() / 255.0
        x += torch.normal(0, std, size=x.shape)
        x = torch.clamp(x, 0, 1)

        return x


class GaussianBlur(T.GaussianBlur):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if random.random() < 0.5:
            return x

        x = super().forward(x)
        return x


class ColorJitter(T.ColorJitter):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if random.random() < 0.5:
            return x

        x = super().forward(x)
        return x


class Resize(T.Resize):
    def __init__(self,*args, **kwargs):
        self.mask_scale = kwargs.pop('mask_scale', 1)
        super().__init__(*args, **kwargs)


class SegTrans(object):
    def __init__(self, transform):
        self.transform = transform
        self.image_only_transforms = (ColorJitter, T.ColorJitter, T.Normalize,
                                      RandomNoise, GaussianBlur, T.GaussianBlur)

    def __call__(self, img, dst):
        if isinstance(self.transform, T.Compose):
            for t in self.transform.transforms:
                img, dst = t(img, dst)

        elif any([isinstance(self.transform, t) for t in self.image_only_transforms]):
            img, dst = self.transform(img), dst

        elif isinstance(self.transform, T.RandomAffine):
            img_size = F._get_image_size(img)
            ret = self.transform.get_params(self.transform.degrees,
                                            self.transform.translate,
                                            self.transform.scale, self.transform.shear, img_size)

            img = F.affine(img, *ret, interpolation=T.InterpolationMode.BILINEAR, fill=self.transform.fill)
            dst = F.affine(dst, *ret, interpolation=T.InterpolationMode.NEAREST, fill=self.transform.fill)

        elif isinstance(self.transform, T.RandomHorizontalFlip):
            if torch.rand(1) < self.transform.p:
                img, dst = F.hflip(img), F.hflip(dst)

        elif isinstance(self.transform, T.RandomVerticalFlip):
            if torch.rand(1) < self.transform.p:
                img, dst = F.vflip(img), F.vflip(dst)

        elif isinstance(self.transform, T.Resize):
            w, h = self.transform.size
            scale = self.transform.mask_scale if isinstance(self.transform, Resize) else 1
            dst = F.resize(dst, [w // scale, h // scale], interpolation=T.InterpolationMode.NEAREST)
            img = self.transform(img)
        else:
            img, dst = self.transform(img), self.transform(dst)
        return img, dst
