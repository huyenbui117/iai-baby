from typing import List
import glob
import os
import random
import torch
from .baby_datamodule import BabyLazyLoadDataset
from .baby_segment_datamodule import BabySegmentDataModule


class BabyLazyLoadSegmentWithHeadMaskDataset(BabyLazyLoadDataset):
    def __init__(self, 
        head_label_paths: List[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.head_label_paths = head_label_paths


    def __getitem__(self, idx):
        img = self.data_module_obj.read_image(self.img_paths[idx], greyscale=self.greyscale)
        label = self.data_module_obj.read_label(self.label_paths[idx])
        label_head = self.data_module_obj.read_head_label(self.head_label_paths[idx])

        if not self.augment:
            return (img, label, label_head)
        
        augmented_tensors = self.data_module_obj.augment_tensors(
            img, torch.cat([label, label_head], dim=0)
        )

        augmented_tensor = random.choice(augmented_tensors)

        augmented_img = augmented_tensor[0]
        augmented_label = augmented_tensor[1][0:1]
        augmented_label_head = augmented_tensor[1][1:]

        return augmented_img, augmented_label, augmented_label_head
        


class BabySegmentWithHeadMaskDataModule(BabySegmentDataModule):
    """LightningDataModule for Baby dataset, with point detection.
    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(
            **kwargs
        )


    def load_data_from_dir(self, data_dir, greyscale=False, augment=False, lazy_load=False):
        img_paths = glob.glob(os.path.join(data_dir, "images/*"))
        label_paths = glob.glob(os.path.join(data_dir, "label/*"))
        head_label_paths = glob.glob(os.path.join(data_dir, "head_label_binary/*"))

        if lazy_load:
            return BabyLazyLoadSegmentWithHeadMaskDataset(
                img_paths=img_paths, 
                label_paths=label_paths, 
                head_label_paths=head_label_paths,
                augment=augment, 
                data_module_obj=self, 
                greyscale=greyscale,
                pred_boxes_path=self.pred_boxes_path
            )
        else:
            raise("Not implemented")