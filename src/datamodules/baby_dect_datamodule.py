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


class BabyDectDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

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
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        image_max_size: Tuple[int, int] = (960, 1728),
        white_pixel: Tuple[int, int, int, int] = (253, 231, 36, 255)
    ):
        super().__init__()

        # this line allows to access init params with 'self.hpimage_max_siarams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        print(type(self.transforms))
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass


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


    def read_image(self, img_path, greyscale=False):
        """
        Read image from path as rgb
        Return tensor(3 x w x h): RGB image tensor
        """
        image = Image.open(img_path)
        
        if greyscale:
            image = ImageOps.grayscale(image)

        transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Pad(self.get_pad_sequence(image.width, image.height, self.hparams.image_max_size[1], self.hparams.image_max_size[0]))
        ])

        img_tensor = transform(image)
        
        return img_tensor


    def read_label(self, img_path):
        """ Read label image and convert to greyscale
        
        Return tensor(1 x w x h): Greyscale image tensor
        """
        img_tensor = self.read_image(img_path)
        
        label_tensor = torch.all(img_tensor.permute(1,2,0) == torch.tensor(self.hparams.white_pixel), dim=-1).unsqueeze(0)

        return label_tensor


    def load_data_from_dir(self, data_dir, convert_x_greyscale=False):
        """Load data from directory

        This method load images from directory and return data as sequence.
        Used for baby data loading
        Return TensorDataset[tuple(x, y1, y2)]
            - x: image
            - y1: segmentation mask
            - y2: centroid detection point
        """
        X, y1, y2 = [], [], []

        path_to_coords = {}

        with open(os.path.join(data_dir, "central_mass.txt")) as mass_file:
            reader = csv.reader(mass_file, delimiter=",")
            next(reader, None)  # skip the headers
            for row in reader:
                file_id = row[0].split("/")[-1]
                path_to_coords[file_id] = torch.tensor([float(row[1]), float(row[2])])

        for path in tqdm.tqdm(glob.glob(os.path.join(data_dir, "images/*")), desc=f"Loading images from {data_dir}"):
            X.append(self.read_image(path, convert_x_greyscale))
        
        for path in tqdm.tqdm(glob.glob(os.path.join(data_dir, "label/*")), desc=f"Loading labels from {data_dir}"):
            y1.append(self.read_label(path))
            file_id = path.split("/")[-1]
            y2.append(path_to_coords[file_id])

        return torch.utils.data.TensorDataset(torch.stack(X), torch.stack(y1), torch.stack(y2))
        # return torch.utils.data.TensorDataset(torch.stack(X).float(), torch.stack(y).float())


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        print("Setup baby", self.hparams.data_dir)

        self.data_train = self.load_data_from_dir(os.path.join(self.hparams.data_dir, "train"))
        self.data_val = self.load_data_from_dir(os.path.join(self.hparams.data_dir, "val"))
        self.data_test = self.load_data_from_dir(os.path.join(self.hparams.data_dir, "test"), convert_x_greyscale=True)

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


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
