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
from .baby_segment_datamodule import BabySegmentDataModule


class BabyLazyLoadLocalizedDataset(BabyLazyLoadDataset):
    def __init__(self, 
        *args, 
        loosen_amount=.1, 
        padding_relative_to_label=False,
        resize_localized_region=(320, 544),
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loosen_amount = loosen_amount
        self.padding_relative_to_label = padding_relative_to_label
        self.resize_localized_region = resize_localized_region

    def calculate_bboxes(self, 
        label_masks: torch.Tensor, 
        loosen_amount: float = 0.03, 
        padding_relative_to_label: bool = False
    ):
        """
        Return the bounding boxes of tensor
        - label_masks: input tensor List(1 x h x w)
        - loosen_amount: float - The rate of the padding of the bounding box
        - padding_relative_to_label: bool - Determine if the loosen amount (padding) is relative to the image size or the label size
        Return:
        - bboxes: List[Tuple(float, float, float, float)] - List of bounding boxes (center_r, center_c, w, h) for all batches in the tensor
        """
        bboxes = []
        for t in label_masks:
            index = (t == 1).nonzero()
                
            rmax_id = torch.argmax(index[:,-2])
            cmax_id = torch.argmax(index[:,-1])
            rmin_id = torch.argmin(index[:,-2])
            cmin_id = torch.argmin(index[:,-1])
            
            r0, c0 = index[rmin_id,1], index[cmin_id, 2]
            r1, c1 = index[rmax_id,1], index[cmax_id, 2]

            padding_w, padding_h = loosen_amount, loosen_amount
            if padding_relative_to_label:
                region_w = c1-c0+1
                region_h = r1-r0+1
                padding_w = region_w / t.shape[-1] * loosen_amount
                padding_h = region_h / t.shape[-2] * loosen_amount

            r0 = max(r0 / t.shape[-2] - padding_h, torch.tensor(0))
            r1 = min(r1 / t.shape[-2] + padding_h, torch.tensor(1))
            c0 = max(c0 / t.shape[-1] - padding_w, torch.tensor(0))
            c1 = min(c1 / t.shape[-1] + padding_w, torch.tensor(1))
            bboxes.append(torch.stack((r0, c0, r1, c1)))

        return bboxes


    def __getitem__(self, idx):
        img = self.data_module_obj.read_image(self.img_paths[idx], greyscale=self.greyscale)
        label = self.data_module_obj.read_label(self.label_paths[idx])
        bboxes = self.calculate_bboxes([label], loosen_amount=self.loosen_amount, padding_relative_to_label=self.padding_relative_to_label)
        r0, c0, r1, c1 = bboxes[0]

        r0 = int(r0 * img.shape[-2])
        r1 = int(r1 * img.shape[-2])
        c0 = int(c0 * img.shape[-1])
        c1 = int(c1 * img.shape[-1])

        img = img[:, r0:r1+1, c0:c1+1]
        label = label[:, r0:r1+1, c0:c1+1]

        transform = transforms.Resize(
            self.resize_localized_region, 
            interpolation=transforms.InterpolationMode.NEAREST
        )
        img, label = transform(img), transform(label)

        if not self.augment:
            return (img, label)

        # augmented_tensors = self.data_module_obj.augment_tensors(torch.stack([img, label]))
        augmented_tensors = self.data_module_obj.augment_tensors(img, label)

        augmented_tensor = random.choice(augmented_tensors)

        augmented_img = augmented_tensor[0]
        augmented_label = augmented_tensor[1]

        return augmented_img, augmented_label

class BabyLazyLoadLocalizedNTAndHeadDataset(BabyLazyLoadLocalizedDataset):
    def __init__(self, 
        *args,
        head_label_paths: List[str] = None,
        resize_localized_region: Tuple[int,int] = (544, 896),
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.head_label_paths = head_label_paths
        self.resize_localized_region = resize_localized_region


    def __getitem__(self, idx):
        img = self.data_module_obj.read_image(self.img_paths[idx], greyscale=self.greyscale)
        label = self.data_module_obj.read_label(self.label_paths[idx])
        label_head = self.data_module_obj.read_head_label(self.head_label_paths[idx])

        bboxes = self.calculate_bboxes([torch.bitwise_or(label.bool(), label_head.bool())], loosen_amount=self.loosen_amount, padding_relative_to_label=self.padding_relative_to_label)
        r0, c0, r1, c1 = bboxes[0]

        r0 = int(r0 * img.shape[-2])
        r1 = int(r1 * img.shape[-2])
        c0 = int(c0 * img.shape[-1])
        c1 = int(c1 * img.shape[-1])

        img = img[:, r0:r1+1, c0:c1+1]
        label = label[:, r0:r1+1, c0:c1+1]

        transform = transforms.Resize(
            self.resize_localized_region, 
            interpolation=transforms.InterpolationMode.NEAREST
        )
        img, label = transform(img), transform(label)

        if not self.augment:
            return (img, label)

        # augmented_tensors = self.data_module_obj.augment_tensors(torch.stack([img, label]))
        augmented_tensors = self.data_module_obj.augment_tensors(img, label)

        augmented_tensor = random.choice(augmented_tensors)

        augmented_img = augmented_tensor[0]
        augmented_label = augmented_tensor[1]

        return augmented_img, augmented_label

class BabySegmentLocalizedDataModule(BabySegmentDataModule):
    """LightningDataModule for Baby dataset, with point detection.
    """

    def __init__(
        self,
        loosen_amount: float = 0.1,
        padding_relative_to_label: bool = False,
        resize_localized_region: Tuple[int,int] = (320, 544),
        **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self.loosen_amount = loosen_amount
        self.padding_relative_to_label = padding_relative_to_label
        self.resize_localized_region = resize_localized_region

    def load_data_from_dir(self, data_dir, greyscale=False, augment=False, lazy_load=True):
        """Load data from directory

        This method load images from directory and return data as sequence.
        Used for baby data loading
        Return TensorDataset[tuple(x, y)]
        """
        img_paths, label_paths = self.get_img_paths(data_dir)

        return BabyLazyLoadLocalizedDataset(
            img_paths, 
            label_paths, 
            augment=augment, 
            data_module_obj=self, 
            greyscale=greyscale,
            loosen_amount=self.loosen_amount,
            padding_relative_to_label=self.padding_relative_to_label
        )


class BabySegmentLocalizedNTAndHeadDataModule(BabySegmentLocalizedDataModule):
    def load_data_from_dir(self, data_dir, greyscale=False, augment=False, lazy_load=True):
        img_paths, label_paths = self.get_img_paths(data_dir)
        head_label_paths = self.get_head_paths(data_dir)

        return BabyLazyLoadLocalizedNTAndHeadDataset(
            img_paths, 
            label_paths, 
            head_label_paths=head_label_paths,
            augment=augment, 
            data_module_obj=self, 
            greyscale=greyscale,
            loosen_amount=self.loosen_amount,
            padding_relative_to_label=self.padding_relative_to_label
        )


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
