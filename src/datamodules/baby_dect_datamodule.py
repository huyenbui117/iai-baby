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

from .baby_datamodule import BabyDataModule, BabyLazyLoadDataset, BabyTupleDataset

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
        lazy_load: bool = True
    ):
        super().__init__(
            data_dir=data_dir,
            image_preprocessor=image_preprocessor,
            label_preprocessor=label_preprocessor,
            augmentations=augmentations,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            lazy_load=lazy_load
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


    def load_data_from_dir(self, data_dir, greyscale=False, augment=False, lazy_load=True):
        """Load data from directory

        This method load images from directory and return data as sequence.
        Used for baby data loading
        Return TensorDataset[tuple(x, y)]
        """

        img_paths = glob.glob(os.path.join(data_dir, "images/*"))
        label_paths = glob.glob(os.path.join(data_dir, "label/*"))

        if lazy_load:
            if augment:
                return BabyLazyLoadDataset(img_paths, label_paths, augment=augment, data_module_obj=self, greyscale=greyscale)
            else:
                return BabyLazyLoadDataset(img_paths, label_paths, augment=augment, data_module_obj=self, greyscale=greyscale)
        else:
            X, y = [], []
            for path in tqdm.tqdm(img_paths, desc=f"Loading images from {data_dir}"):
                X.append(self.read_image(path, greyscale))
            
            for path in tqdm.tqdm(label_paths, desc=f"Loading labels from {data_dir}"):
                y.append(self.read_label(path))

            if augment:
                augmented_X, augmented_y = [], []
                for idx, x in enumerate(X):
                    augmented_tensors = self.augment_tensors(torch.stack((x, y[idx])))
                    for augmented_tensor in augmented_tensors:
                        augmented_X.append(augmented_tensor[0])
                        augmented_y.append(augmented_tensor[1])
                
                return BabyTupleDataset(augmented_X, augmented_y)        
            else:
                return BabyTupleDataset(X, y)