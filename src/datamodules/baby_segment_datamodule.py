from typing import Any, Dict, Optional, Tuple, List, Union

import glob
import random
import os
from PIL import Image, ImageOps
import tqdm
import gc

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from .baby_datamodule import BabyDataModule, BabyTupleDataset, BabyLazyLoadDataset



class BabySegmentDataModule(BabyDataModule):
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
        resize_input: Union[Tuple[int, int], None] = None,
        white_pixel: Tuple[int, int, int, int] = (253, 231, 36, 255),
        lazy_load: bool = False,
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
            resize_input=resize_input,
            white_pixel=white_pixel,
            lazy_load=lazy_load,
        )


    def load_data_from_dir(self, data_dir, greyscale=False, augment=False, lazy_load=False):
        """Load data from directory

        This method load images from directory and return data as sequence.
        Used for baby data loading
        Return TensorDataset[tuple(x, y)]
        """

        img_paths = glob.glob(os.path.join(data_dir, "images/*"))
        label_paths = glob.glob(os.path.join(data_dir, "label/*"))

        if lazy_load:
            return BabyLazyLoadDataset(img_paths, label_paths, augment=augment, data_module_obj=self, greyscale=greyscale)
        else:
            X, y = [], []
            for path in tqdm.tqdm(img_paths, desc=f"Loading images from {data_dir}"):
                _x = self.read_image(path, greyscale)
                _x = self.image_preprocessor(_x)
                X.append(_x)
            
            for path in tqdm.tqdm(label_paths, desc=f"Loading labels from {data_dir}"):
                _y = self.read_label(path)
                _y = self.label_preprocessor(_y)
                y.append(_y)

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


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
