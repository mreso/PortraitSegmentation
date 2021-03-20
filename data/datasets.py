from abc import abstractmethod
from os import path as osp

import numpy as np
import torch
from PIL import Image as Image

from torchvision import transforms as T

from data.transforms import Resize, ColorJitter, GaussianBlur, RandomNoise, SegTrans


class SegmentationDataset(torch.utils.data.Dataset):

    def __init__(self, **config):
        self.transforms = config.get('transforms', None)

    @abstractmethod
    def img_file_path(self, idx):
        raise NotImplemented

    @abstractmethod
    def dst_file_path(self, idx):
        raise NotImplemented

    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        img = Image.open(self.img_file_path(idx)).convert('RGB')

        dst = Image.open(self.dst_file_path(idx)).convert('L')

        thr = np.zeros_like(dst, dtype=np.uint8)
        thr[np.array(dst) > 200] = 255

        img, label = self.transforms(img, Image.fromarray(thr))

        return img, label

    def __len__(self):
        return len(self.filenames)


class NukkiDataset(SegmentationDataset):
    def __init__(self, root_folder, mode='train', **config):
        self.root_folder = root_folder
        self.transforms = config.get('transforms', None)
        self.img_folder = config.get('img_folder', 'input')
        self.dst_folder = config.get('dst_folder', 'target')
        self.mode = mode
        super().__init__(**config)

        txt_file = osp.join(root_folder, mode + '.txt')
        assert osp.exists(txt_file)

        with open(txt_file) as f:
            self.filenames = [_.strip() for _ in f.readlines()]

    def img_file_path(self, idx):
        return osp.join(*(self.root_folder, self.img_folder, self.filenames[idx]))

    def dst_file_path(self, idx):
        return osp.join(*(self.root_folder, self.dst_folder, self.filenames[idx]))


class EG1800Dataset(SegmentationDataset):
    def __init__(self, root_folder, mode='train', **config):
        self.root_folder = root_folder
        self.transforms = config.get('transforms', None)
        self.img_folder = config.get('img_folder', 'images_data_crop')
        self.dst_folder = config.get('dst_folder', 'GT_png')
        self.mode = mode
        super().__init__(**config)

        txt_file = osp.join(root_folder, f'EG1800_{mode}.txt')
        assert osp.exists(txt_file)

        with open(txt_file) as f:
            self.filenames = [_.strip() for _ in f.readlines()]

    def img_file_path(self, idx):
        return osp.join(*(self.root_folder, self.img_folder, f'{self.filenames[idx]}.jpg'))

    def dst_file_path(self, idx):
        return osp.join(*(self.root_folder, self.dst_folder, f'{self.filenames[idx]}_mask.png'))


class ConcatenatedSegmentationDataset(SegmentationDataset):
    def __init__(self, datasets: list, **config):
        self.ranges = []
        self.datasets = []
        for ds in datasets:
            if len(self.ranges) == 0:
                self.ranges.append(range(len(ds)))
            else:
                self.ranges.append(range(self.ranges[-1][-1] + 1, self.ranges[-1][-1] + len(ds) + 1))
            self.datasets.append(ds)
        super().__init__(**config)

    def img_file_path(self, idx):
        ds_idx = [idx in _ for _ in self.ranges].index(True)
        return self.datasets[ds_idx].img_file_path(idx - self.ranges[ds_idx][0])

    def dst_file_path(self, idx):
        ds_idx = [idx in _ for _ in self.ranges].index(True)
        return self.datasets[ds_idx].dst_file_path(idx - self.ranges[ds_idx][0])

    def __len__(self):
        return self.ranges[-1][-1] + 1

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def determine_mean_and_std(dataset):
    print('Re-computing mean of dataset')
    mean = np.zeros((1, 3))
    std = np.zeros((1, 3))

    for img, _ in dataset:
        mean += np.mean(np.array(img), axis=(0, 1))
        std += np.std(np.array(img), axis=(0, 1))

    mean /= len(dataset)
    std /= len(dataset)
    return mean, std


def create_datasets(config):
    image_size = config.get('image_size', (224, 224))

    train_dataset = ConcatenatedSegmentationDataset([
        EG1800Dataset('datasets/EG1800/', config=config),
        NukkiDataset('datasets/Nukki/baidu_V1/', config=config),
        NukkiDataset('datasets/Nukki/baidu_V2/', config=config),
    ])

    data_info_path = 'datasets/data_info.pt'
    if osp.exists(data_info_path):
        data_info = torch.load(data_info_path)
    else:
        mean, std = determine_mean_and_std(train_dataset)
        data_info = {'mean': mean.T, 'std': std.T}
        torch.save(data_info, 'datasets/data_info.pt')

    train_transforms = (Resize(image_size, mask_scale=config.get('mask_scale', 1)),
                        T.RandomAffine(45, scale=(0.5, 1.5)),
                        T.RandomHorizontalFlip(),
                        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                        GaussianBlur(5),
                        T.ToTensor(),
                        RandomNoise(scale=10.0),
                        T.Normalize(data_info['mean'].reshape((3, 1, 1)) / 255.,
                                    data_info['std'].reshape((3, 1, 1)) / 255.),
                        )
    train_transforms = SegTrans(T.Compose([SegTrans(m) for m in train_transforms]))

    train_dataset.transforms = train_transforms

    val_transforms = (
        Resize(image_size, mask_scale=config.get('mask_scale', 1)),
        T.ToTensor(),
        T.Normalize(data_info['mean'].reshape((3, 1, 1)) / 255., data_info['std'].reshape((3, 1, 1)) / 255.),
    )

    val_transforms = SegTrans(T.Compose([SegTrans(m) for m in val_transforms]))

    val_dataset = ConcatenatedSegmentationDataset([
        # NukkiDataset('datasets/Nukki/baidu_V1/', mode='val'),
        # NukkiDataset('datasets/Nukki/baidu_V2/', mode='val'),
        EG1800Dataset('datasets/EG1800/', mode='val'),
    ], transforms=val_transforms)

    return {'train': train_dataset, 'val': val_dataset}
