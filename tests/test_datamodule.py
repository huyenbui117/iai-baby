from pathlib import Path

import pytest
import torch

from src.datamodules.baby_segment_datamodule import BabySegmentDataModule
from src.models.preprocess.CustomAlbumentation import AlbumentationWrapper
import albumentations

@pytest.mark.parametrize("batch_size", [4])
def test_baby_datamodule(batch_size):
    data_dir = "data/"

    preprocessor = AlbumentationWrapper(
        transform=albumentations.Resize(
            height=320,
            width=544
        )
    )
    dm = BabySegmentDataModule(
        data_dir=data_dir, 
        batch_size=batch_size,
        wandb_project="baby",
        wandb_artifact="baby-team/baby/baby:latest",
        image_preprocessor=preprocessor,
        label_preprocessor=preprocessor,
    )
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "baby").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.test_dataloader()))
    x, y, *args = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert x.shape == y.shape
    print(y)
    print(x.shape, y.shape)
    print(y.unique())
    print(x.unique())
    assert len(y.unique()) == 2
