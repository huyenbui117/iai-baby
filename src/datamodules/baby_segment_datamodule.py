from typing import Any, Dict, Optional, Tuple

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


    def load_data_from_dir(self, data_dir, convert_x_greyscale=False, augment=False):
        """Load data from directory

        This method load images from directory and return data as sequence.
        Used for baby data loading
        Return TensorDataset[tuple(x, y)]
        """
        X, y = [], []
        for path in tqdm.tqdm(glob.glob(os.path.join(data_dir, "images/*")), desc=f"Loading images from {data_dir}"):
            X.append(self.read_image(path, convert_x_greyscale)/255)
        
        for path in tqdm.tqdm(glob.glob(os.path.join(data_dir, "label/*")), desc=f"Loading labels from {data_dir}"):
            y.append(self.read_label(path))

        if augment:
            # X_tensor = torch.stack(X)
            # y_tensor = torch.stack(y)
            # augmented_tensors = self.augment_tensors(torch.stack([X_tensor, y_tensor]))
            # augmented_tensors = augmented_tensors.view((
            #     augmented_tensors.shape[1], 
            #     augmented_tensors.shape[0],
            #     *augmented_tensors.shape[2:]
            # ))
            # augmented_X = augmented_tensors[0]
            # augmented_X = augmented_X.view((
            #     augmented_X.shape[0]*augmented_X.shape[1],
            #     *augmented_X.shape[2:]
            # ))
            # augmented_y = augmented_tensors[1]
            # augmented_y = augmented_y.view((
            #     augmented_y.shape[0]*augmented_y.shape[1],
            #     *augmented_y.shape[2:]
            # ))
            # return torch.utils.data.TensorDataset(augmented_X, augmented_y)
            augmented_X, augmented_y = [], []
            for idx, x in enumerate(X):
                augmented_tensors = self.augment_tensors(torch.stack((x, y[idx])))
                for augmented_tensor in augmented_tensors:
                    augmented_X.append(augmented_tensor[0])
                    augmented_y.append(augmented_tensor[1])
            # import IPython ; IPython.embed()
            
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
