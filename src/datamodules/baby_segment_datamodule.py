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
        pred_boxes_path: Union[str, None] = None,
        **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self.pred_boxes_path = pred_boxes_path

    def get_img_paths(self, data_dir):
        img_paths = glob.glob(os.path.join(data_dir, "images/*"))
        label_paths = glob.glob(os.path.join(data_dir, "label/*"))
        return img_paths, label_paths


    def load_data_from_dir(self, data_dir, greyscale=False, augment=False, lazy_load=False):
        """Load data from directory

        This method load images from directory and return data as sequence.
        Used for baby data loading
        Return TensorDataset[tuple(x, y)]
        """

        img_paths, label_paths = self.get_img_paths(data_dir)

        if lazy_load:
            return BabyLazyLoadDataset(
                img_paths, 
                label_paths, 
                augment=augment, 
                data_module_obj=self, 
                greyscale=greyscale,
                pred_boxes_path=self.pred_boxes_path
            )
        else:
            raise("Not implemented")


class BabyFullBodySegmentDataModule(BabySegmentDataModule):
    def __init__(
        self,
        **kwargs
    ):
        """Segment data module for full body images

        Args:
            fullbody_img_names (str, optional): Path to file that contain full body image names. Defaults to None.
        """
        super().__init__(
            **kwargs
        )

    def get_img_paths(self, data_dir):

        img_paths = glob.glob(os.path.join(data_dir, "images/*"))
        label_paths = glob.glob(os.path.join(data_dir, "label/*"))

        with open(os.path.join(data_dir, "fullbody_imgs.txt")) as f:
            fullbody_names = f.read().split("\n")
            if fullbody_names[-1] == "":
                fullbody_names.pop()
        img_paths = [p for p in img_paths if os.path.basename(p) in fullbody_names]
        label_paths = [p for p in label_paths if os.path.basename(p) in fullbody_names]
        assert len(img_paths) == len(label_paths)

        return img_paths, label_paths


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
