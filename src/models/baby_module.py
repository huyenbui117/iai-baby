from typing import Any, List, Union, Callable
import os
import random
import shutil
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, F1Score, Accuracy, Precision, Recall, JaccardIndex
import segmentation_models_pytorch
from torchvision.utils import save_image

class BabyLitModule(LightningModule):
    """Example of LightningModule for Baby dataset.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: segmentation_models_pytorch.Unet = None,
        optimizer: torch.optim.Optimizer = None,
        postprocessor: Union[Callable, None] = None,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        lr_scheduler_monitor: str = None,
        loss_func: torch.nn.CrossEntropyLoss = None,
        eval_img_path: str = "./tmp",
        log_train_img: float = 0.2,
        log_val_img: float = 1.,
        log_test_img: float = 1.,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.log_train_img = log_train_img
        self.log_val_img = log_val_img
        self.log_test_img = log_test_img
        
        self.net = net
        self.postprocessor = postprocessor
        
        # loss function
        # self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weight))
        self.criterion = loss_func

        # for logging val image
        shutil.rmtree(self.hparams.eval_img_path, ignore_errors=True)
        os.makedirs(self.hparams.eval_img_path, exist_ok=True)


    def forward(self, x: torch.Tensor):
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        img_log_paths = self.log_batch_img(batch, batch_idx, preds, targets, phase="train") if self.log_train_img else []

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {
            "loss": loss,
            "img_log_paths": img_log_paths
        }

    
    def log_images(self, 
        log_key: str, 
        outputs: List[Any], 
        log_ratio: float = .2
    ):
        # `outputs` is a list of dicts returned from `training_step()`
        img_log_paths = [path for output in outputs for path in output["img_log_paths"]]
        if len(img_log_paths) > 0 and self.logger is not None and hasattr(self.logger, "log_image"):
            if log_ratio is None or log_ratio >= 1:
                # log all images
                self.logger.log_image(log_key, img_log_paths)
            else:
                div_factor = int(1/log_ratio)
                self.logger.log_image(log_key, [path for idx, path in enumerate(img_log_paths) if idx % div_factor == 0])

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        img_log_paths = self.log_batch_img(batch, batch_idx, preds, targets, phase="val") if self.log_val_img else []

        return {
            "loss": loss,
            "img_log_paths": img_log_paths
        }


    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True)

        img_log_paths = self.log_batch_img(batch, batch_idx, preds, targets, phase="test") if self.log_test_img else []

        return {
            "loss": loss,
            "img_log_paths": img_log_paths
        }


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        optimizer = self.hparams.optimizer(params=self.parameters())
        lr_scheduler = {
            "scheduler": self.hparams.lr_scheduler(optimizer=optimizer),
            "monitor": self.hparams.lr_scheduler_monitor
        } if self.hparams.lr_scheduler else None

        return {
            "optimizer": optimizer,
        } if lr_scheduler is None else {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }
