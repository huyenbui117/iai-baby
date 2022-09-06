from typing import Any, Dict, Optional, Tuple

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

from .baby_datamodule import BabyDataModule

class BabyTupleDataset(torch.utils.data.Dataset):
    def __init__(self, *tuples: Tuple[torch.Tensor]):
        """
        tuples: tuple of tensors (channel x h x w)
        """
        assert all(len(tuples[0]) == len(t) for t in tuples), "Size mismatch between tensors"
        self.tuples = tuples

    def __getitem__(self, idx):
        return tuple(t[idx] for t in self.tuples)

    def __len__(self):
        return len(self.tuples[0])


class BabyLazyLoadDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, label_paths, data_module_obj=None, greyscale=False):
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.data_module_obj = data_module_obj
        self.greyscale = greyscale
    
    def __getitem__(self, idx):
        img = self.data_module_obj.read_image(self.img_paths[idx], greyscale=self.greyscale)/255
        label = self.data_module_obj.read_label(self.label_paths[idx])
        
        augmented_imgs = self.data_module_obj.augment_tensors(img)
        augmented_labels = self.data_module_obj.augment_tensors(label)

        return (img, label)
        return (random.choice(augmented_imgs), random.choice(augmented_labels))


    def __len__(self):
        return len(self.img_paths)


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
            if augment:
                return BabyLazyLoadDataset(img_paths, label_paths, data_module_obj=self, greyscale=greyscale)
            else:
                return BabyLazyLoadDataset(img_paths, label_paths, data_module_obj=self, greyscale=greyscale)
        else:
            X, y = [], []
            for path in tqdm.tqdm(img_paths, desc=f"Loading images from {data_dir}"):
                X.append(self.read_image(path, greyscale)/255)
            
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


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
