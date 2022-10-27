from typing import Any, Dict, Optional, Tuple, List, Union

import json
from collections import defaultdict
import os
import wandb
from PIL import Image, ImageOps
import random
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms


class BabyDataModule(LightningDataModule):
    """LightningDataModule for Baby dataset, with point detection.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        image_preprocessor: transforms.Compose=transforms.Compose([]),
        label_preprocessor: transforms.Compose=transforms.Compose([]),
        augmentations: List[List[transforms.Compose]]=[[]],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        padding: bool = True,
        wandb_project: str = "baby",
        wandb_artifact: str = "baby-team/baby/baby:latest",
        lazy_load: bool = True,
    ):
        super().__init__()

        self.data_dir = data_dir

        # this line allows to access init params wit[transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]h 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.image_preprocessor = image_preprocessor
        self.label_preprocessor = label_preprocessor
        self.augmentations = augmentations
        self.lazy_load = lazy_load
        self.padding = padding

        self.wandb_project = wandb_project
        self.wandb_artifact = wandb_artifact

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.prepare_data()


    def prepare_data(self):
        """Download data if needed.
        """
        api = wandb.Api()
        artifact = api.artifact(self.wandb_artifact, type="dataset")
        artifact.download(root=self.data_dir)


    def get_pad_sequence(self, img_width, img_height, pad_width, pad_height):
        """
        Return the pad sequence needed to pad from (img_width, img_height) -> (pad_width, pad_height)
        """
        horizontal, vertical = pad_width-img_width, pad_height-img_height
        
        l = horizontal // 2
        r = l + horizontal % 2
        
        u = vertical // 2
        d = u + vertical % 2
        
        pad_sequence = [l, u, r, d]
        return pad_sequence

    def read_image(self, 
        img_path, 
        greyscale=False, 
        scale_factor=255,
        preprocess=True
    ):
        """
        Read image from path as rgb
        Return tensor(1 x w x h): RGB image tensor
        """
        image = Image.open(img_path)
        
        if greyscale:
            image = ImageOps.grayscale(image)

        img_tensor = transforms.PILToTensor()(image)
        if preprocess:
            img_tensor = self.image_preprocessor(img_tensor)

        return img_tensor/scale_factor


    def read_head_label(self, img_path, preprocess=True):
        label_tensor = self.read_image(img_path, scale_factor=255., preprocess=False)
        if preprocess:
            label_tensor = self.label_preprocessor(label_tensor)
        return label_tensor


    def read_label(self, img_path, preprocess=True):
        """ Read label image and convert to greyscale
        
        Return tensor(1 x w x h): Greyscale image tensor
        """
        label_tensor = self.read_image(img_path, scale_factor=255., preprocess=False)
        if preprocess:
            label_tensor = self.label_preprocessor(label_tensor)

        return label_tensor


    def augment_tensors(self, img_tensor: torch.Tensor, label_tensor: torch.Tensor):
        """
        img_tensor: Tuple of tensors shaped (channel x h x w)
        label_tensor: Tuple of tensors shaped (channel x h x w)
        Return: List[(augmented_img_tensor, augmented_label_tensor)] 
        list of tensors augmented from tensors
        """
        if len(self.augmentations) == 0:
            return [(img_tensor, label_tensor)]
        
        augmented_tensors = []

        for transforms in self.augmentations:
            aug_img_tensor, aug_label_tensor = img_tensor.clone(), label_tensor.clone()
            for transform, apply_to_image, apply_to_label in transforms:
                if apply_to_image and apply_to_label: 
                    img_idx = aug_img_tensor.shape[0]
                    concat_tensor = transform(
                        torch.cat((aug_img_tensor, aug_label_tensor), dim=0)
                    )
                    aug_img_tensor, aug_label_tensor = concat_tensor[:img_idx], concat_tensor[img_idx:]
                elif apply_to_image:
                    aug_img_tensor = transform(aug_img_tensor)
                elif apply_to_label:
                    aug_label_tensor = transform(aug_label_tensor)
            augmented_tensors.append((aug_img_tensor, aug_label_tensor))

        return augmented_tensors


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        print("Setup baby", self.hparams.data_dir)

        self.data_train = self.load_data_from_dir(
            os.path.join(self.hparams.data_dir, "train"), 
            augment = True, 
        )
        self.data_val = self.load_data_from_dir(
            os.path.join(self.hparams.data_dir, "val"), 
        )
        self.data_test = self.load_data_from_dir(
            os.path.join(self.hparams.data_dir, "test"), 
            greyscale=True, 
            lazy_load=self.lazy_load
        )
        
        self.loader_train = DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
        self.loader_val = DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        self.loader_test = DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        

    def train_dataloader(self):
        return self.loader_train

    def val_dataloader(self):
        return self.loader_val

    def test_dataloader(self):
        return self.loader_test

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass



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
    def __init__(self, 
        img_paths: List[str], 
        label_paths: List[str], 
        augment: bool = False, 
        data_module_obj: BabyDataModule = None, 
        greyscale: bool = False,
        pred_boxes_path: Union[str, None] = None,
        **kwargs
    ):
        """Dataset for baby

        Args:
            img_paths (List[str]): path of images
            label_paths (List[str]): path of NT mask
            augment (bool, optional): Augment the read images or not. Defaults to False.
            data_module_obj (BabyDataModule, optional): The data module object to use important functions. Defaults to None.
            greyscale (bool, optional): Convert to greyscale or not. Defaults to False.
            pred_boxes_path (Union[str, None], optional): The path to bounding box prediction. Set to true to return localized region of NT. Defaults to None.
        """
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.augment = augment
        self.data_module_obj = data_module_obj
        self.greyscale = greyscale

        self.pred_boxes_path = pred_boxes_path
        if pred_boxes_path is not None:
            self.imgid_to_boxes = defaultdict(lambda: [])
            with open(pred_boxes_path) as fin:
                boxes = json.load(fin)
                for box in boxes:
                    self.imgid_to_boxes[box["image_id"]].append(box)
            for key in self.imgid_to_boxes:
                self.imgid_to_boxes[key] = sorted(
                    self.imgid_to_boxes[key],
                    key=lambda box: box["score"],
                    reverse=True
                )
        

    def __getitem__(self, idx):
        img = self.data_module_obj.read_image(self.img_paths[idx], greyscale=self.greyscale)
        label = self.data_module_obj.read_label(self.label_paths[idx])

        assert img.shape == label.shape
        assert len(label.unique()) == 2

        extras = []

        if self.pred_boxes_path:
            image_id = os.path.basename(self.img_paths[idx]).split(".")[0]
            boxes = self.imgid_to_boxes[image_id]
            extras.extend([
                torch.tensor(boxes[0]["bbox"]),
                torch.tensor(boxes[0]["score"]),
            ])

        if not self.augment:
            return (img, label, *extras)
        
        augmented_tensors = self.data_module_obj.augment_tensors(img, label)

        augmented_tensor = random.choice(augmented_tensors)

        augmented_img = augmented_tensor[0]
        augmented_label = augmented_tensor[1]

        return augmented_img, augmented_label, *extras


    def __len__(self):
        return len(self.img_paths)