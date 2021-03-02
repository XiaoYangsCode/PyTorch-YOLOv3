import glob
import random
import os
import sys
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    """[将非正方形的图片填充为正方形图片]

    Args:
        img : [图片 shape (c,h,w)]
        pad_value : [填充的默认值]
    """
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    """图像差值到相应的尺寸
    Args:
        image : shape (c,h,w)
        size : final size

    Returns:
        image : shape (c,h,w)
    """
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    """
    图片数据集类，定义了数据集的读取方式 和 transform 进行 pad resize 等操作 
    """
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'), 
            dtype=np.uint8)
        
        # img shape (H,W,C)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        # bounding box info path
        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32      # size range for multiscale training
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):
        
        # ---------
        #  Image shape (h,w,c)
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip()

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception as e:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception as e:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform bb_targets shape (-1, 6)   (index,category,x,y,w,h)
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except:
                print(f"Could not apply transform.")
                return

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1
        # batch have transformed
        # Drop invalid images
        batch = [data for data in batch if data is not None]

        # *batch = (path,img,bb_targets) ... 
        # list( zip(*batch) ) = [(path,...), (img,...), (bb_targets,...)]
        paths, imgs, bb_targets = list(zip(*batch))
        
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        
        # Resize images to input shape  (batch,c,h,w)    img (c,h,w)    input imgs is tuple    len = batch
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets shape (-1, 6)    bb_targets is tuple len = batch
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)
        
        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)
