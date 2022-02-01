from torchaudio import transforms
from tqdm import tqdm
import utils
import os
import random
import argparse
import numpy as np

from utils import ext_transforms as et
from datasets import MEDSegmentation

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from datetime import datetime
import socket

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    """ path 지정해줘야 함.
    """
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default="med", 
                        choices=['med'], help='Name of dataset')
    parser.add_argument("--num_clases", type=int, default=2,
                        help="num class (default: 2")
    parser.add_argument("--version", type=str, default='220131',
                        choices=['220131'], help='version of MED')
    parser.add_argument("--crop_size", type=int, default=513)

    return parser

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'med':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]), 
        ])

    return train_transform

def main():
    
    rootDir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    opts = get_argparser().parse_args()
    root = os.path.expanduser(opts.data_root)
    imgdir = os.path.join(rootDir, root, 'test/images/I0000647.bmp')
    maskdir = os.path.join(rootDir, root, 'test/masks/I0000647_mask.bmp')

    img = Image.open(imgdir).convert('RGB')
    target = Image.open(maskdir)
    transform = get_dataset(opts=opts)

    tmp = et.ExtRandomScale((0.5, 2.0))
    img, target = tmp(img, target)
    print(img._size, target._size)
    #if transform is not None:
    #    img, target = transform(img, target)

    

if __name__ == '__main__':
    main()