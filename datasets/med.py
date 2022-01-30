import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np

from PIL import Image

DATASET_VER_DICT = {
    '220131': {
        'filename': 'MEDtrainval_31-JAN-2022',
        'base_dir': 'MEDdevkit/MED220131',
        'md5': 'TBD'
    },
}

def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class MEDSegmentation(data.Dataset):
    """
    Args:
        root (string): Root directory of the MED Dataset.
        version (string, optional): The dataset version, supports version TBD.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    cmap = voc_cmap()
    def __init__(self, root, version='220131', image_set='train', transform=None):

        is_aug=False
        
        self.root = os.path.expanduser(root)
        # ./datasets/data
        self.version = version
        # 20220131
        self.filename = DATASET_VER_DICT[version]['filename']
        # MEDtrainval_31-JAN-2022
        self.transform = transform
        
        self.image_set = image_set
        # train
        base_dir = DATASET_VER_DICT[version]['base_dir']
        # MEDdevkit/MED220131
        med_root = os.path.join(self.root, base_dir)
        # ~/datasets/data/MEDdevkit/MED220131
        image_dir = os.path.join(med_root, 'Images')
        # ~/datasets/data/MEDdevkit/MED220131/Images
        # ... images

        if not os.path.isdir(med_root):
            raise RuntimeError('Dataset not found or corrupted.')
        
        if is_aug and image_set=='train': # is_aug = Flase
            mask_dir = os.path.join(med_root, 'MasksAug')
            # ~/datasets/data/MEDdevkit/MED220131/MasksAug
            assert os.path.exists(mask_dir), "Masks not found, please refer to README.md and prepare it manually"
            split_f = os.path.join( self.root, 'train_aug.txt')
            # ./datasets/data/train_aug.txt
        else:
            mask_dir = os.path.join(med_root, 'Masks')
            # ~/datasets/data/MEDdevkit/MED220131/Masks
            # ... images
            splits_dir = os.path.join(med_root, 'Segmentation/2080')
            # ~/datasets/data/MEDdevkit/MED220131/Segmentation/2080
            # train.txt (ln=...), trainval.txt (ln=...), val.txt (ln=...)
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
            # ~/datasets/data/MEDdevkit/MED220131/Segmentation/2080/train.txt

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
            # str.strip() 공백 제거
            # ex) Exxxx-Ixxxx
        
        self.images = [os.path.join(image_dir, x + ".bmp") for x in file_names]
        # ~/datasets/data/MEDdevkit/MED220131/Images/Exxxx-Ixxxx.bmp
        self.masks = [os.path.join(mask_dir, x + ".bmp") for x in file_names]
        # ~/datasets/data/MEDdevkit/MED220131/Masks/Exxxx-Ixxxx.bmp
        assert (len(self.images) == len(self.masks))
        # ... = ...

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        """ Transform PIL to tensor
        """
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target


    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]