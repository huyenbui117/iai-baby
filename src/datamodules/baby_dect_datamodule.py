from typing import Any, Dict, Optional, Tuple

import csv
import glob
import os
from PIL import Image, ImageOps
import tqdm
import gc

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from .baby_datamodule import BabyDataModule

class BabyDectDataModule(BabyDataModule):
    """LightningDataModule for Baby dataset, with point detection.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        image_preprocessor: transforms.Compose=transforms.Compose([]),
        label_preprocessor: transforms.Compose=transforms.Compose([]),
        augmentations: Tuple[transforms.Compose, ...]=(transforms.Compose([]),),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        image_max_size: Tuple[int, int] = (960, 1728),
        white_pixel: Tuple[int, int, int, int] = (253, 231, 36, 255)
    ):
        super().__init__(
            data_dir=data_dir,
            image_preprocessor=image_preprocessor,
            label_preprocessor=label_preprocessor,
            augmentations=augmentations,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            image_max_size=image_max_size,
            white_pixel=white_pixel
        )


    def calculate_central_mass(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Return the central position point of a label binary map
        - t: input tensor (b x h x w)
        Return tensors: normalized centroid (b x 2)
        """
        rows, cols = [], []
        for t in imgs:
            r, c = 0, 0
            index = (t == 1).nonzero().float()
            
            r = index[:,0].mean()/t.shape[0]
            c = index[:,1].mean()/t.shape[1]
            
            rows.append(r)
            cols.append(c)
        
        rows_tensor = torch.stack(rows)
        cols_tensor = torch.stack(cols)
        return torch.stack((rows_tensor, cols_tensor), dim=-1)


    def load_data_from_dir(self, data_dir, greyscale=False, augment=False):
        """Load data from directory

        This method load images from directory and return data as sequence.
        Used for baby data loading
        Return TensorDataset[tuple(x, y1, y2)]
            - x: image
            - y1: segmentation mask
            - y2: centroid detection point
        """
        X, y1, y2 = [], [], []

        for path in tqdm.tqdm(glob.glob(os.path.join(data_dir, "images/*")), desc=f"Loading images from {data_dir}"):
            _x = self.read_image(path, greyscale).float()/255
            if self.image_preprocessor:
                _x = self.image_preprocessor(_x)
                
            X.append(_x)
        
        for path in tqdm.tqdm(glob.glob(os.path.join(data_dir, "label/*")), desc=f"Loading labels from {data_dir}"):
            _y1 = self.read_label(path)
            if self.label_preprocessor:
                _y1 = self.label_preprocessor(_y1)
            
            y1.append(_y1)
            file_id = path.split("/")[-1]

            # y2.append(path_to_coords[file_id])

        X_tensor = torch.stack(X)
        y1_tensor = torch.stack(y1)

        if augment:
            augmented_tensors = self.augment_tensors(torch.stack([X_tensor, y1_tensor]))
            augmented_X = torch.cat([t[0] for t in augmented_tensors], dim=0)
            augmented_y1 = torch.cat([t[1] for t in augmented_tensors], dim=0)
            augmented_y2 = self.calculate_central_mass(augmented_y1[:,0,:,:])
            return torch.utils.data.TensorDataset(augmented_X, augmented_y1, augmented_y2)
        else:
            y2_tensor = self.calculate_central_mass(y1_tensor[:,0,:,:])
            return torch.utils.data.TensorDataset(X_tensor, y1_tensor, y2_tensor)        
        