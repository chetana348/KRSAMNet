import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import IterableDataset


class ResizeTwoSizes:
    def __init__(self, size1, size2):
        self.resize1 = transforms.Resize((size1, size1), interpolation=InterpolationMode.NEAREST)
        self.resize2 = transforms.Resize((size2, size2), interpolation=InterpolationMode.NEAREST)

    def __call__(self, image):
        return self.resize1(image), self.resize2(image)


class DataGen(Dataset if 'train' in ['train', 'val'] else IterableDataset):
    def __init__(self, image_root, gt_root, size1, size2, mode='train'):
        self.image_paths = sorted([
            os.path.join(image_root, f)
            for f in os.listdir(image_root)
            if f.endswith('.tif') or f.endswith('.png')
        ])
        self.gt_paths = sorted([
            os.path.join(gt_root, f)
            for f in os.listdir(gt_root)
            if f.endswith('.tif') or f.endswith('.png')
        ])

        self.size1 = size1
        self.size2 = size2
        self.mode = mode
        self.length = len(self.image_paths)
        self.index = 0

        self.resize = ResizeTwoSizes(size1, size2)
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def _load_rgb(self, path):
        return Image.open(path).convert('RGB')

    def _load_binary(self, path):
        label = Image.open(path)
        label = np.array(label)
        label = (label > 0).astype(np.uint8) * 255
        return Image.fromarray(label)

    def __getitem__(self, idx):
        if self.mode == 'test':
            raise NotImplementedError("Use iterator style (`for x in dataset`) for test mode.")

        img_path = self.image_paths[idx]
        gt_path = self.gt_paths[idx]

        img = self._load_rgb(img_path)
        label = self._load_binary(gt_path)

        img1, img2 = self.resize(img)
        label, _ = self.resize(label)

        img1 = self.normalize(self.to_tensor(img1))
        img2 = self.normalize(self.to_tensor(img2))
        label = self.to_tensor(label)

        return {'image1': img1, 'image2': img2, 'label': label}

    def __len__(self):
        return self.length

    def __iter__(self):
        if self.mode != 'test':
            raise NotImplementedError("Use indexing (`dataset[idx]`) for train/val mode.")
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration

        img_path = self.image_paths[self.index]
        gt_path = self.gt_paths[self.index]
        name = os.path.basename(img_path)

        img = self._load_rgb(img_path)
        label = self._load_binary(gt_path)

        img1, img2 = self.resize(img)
        label, _ = self.resize(label)

        img1 = self.normalize(self.to_tensor(img1))
        img2 = self.normalize(self.to_tensor(img2))
        label_np = np.array(self.to_tensor(label).squeeze(0))  # convert to numpy mask

        self.index += 1
        return img1.unsqueeze(0), img2.unsqueeze(0), label_np, name

