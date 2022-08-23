from typing import Any, List
import os
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, F1Score, Accuracy, Precision, Recall, JaccardIndex
import segmentation_models_pytorch
from torchvision.utils import save_image

class BabyDectLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

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
        net: segmentation_models_pytorch.Unet,
        optimizer: torch.optim.Optimizer,
        loss_func: torch.nn.CrossEntropyLoss,
        eval_img_path: str = "./tmp",
        log_train_img: bool = True,
        log_val_img: bool = True,
        log_test_img: bool = True,
    ):
        super().__init__()

        MDMC_REDUCE = "samplewise"

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.log_train_img = log_train_img
        self.log_val_img = log_val_img
        self.log_test_img = log_test_img
        
        self.net = net

        # loss function
        # self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weight))
        self.criterion = loss_func

        # for logging val image
        os.makedirs(self.hparams.eval_img_path, exist_ok=True)
        self.log_img_paths = []

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, _, y = batch
        
        preds = self.forward(x.float())

        # y ~ normalized coordinate (x1,y1)
        loss = self.criterion(preds, y)
        return loss, preds, y

    def log_batch_img(self, batch, batch_idx, preds, targets, phase="train", point_size=7):
        # Calculate and log predicted images
        img_tensor = batch[0].swapaxes(0,1) # img_tensor shape: (channel(1) x n x h x w)
        img_mask = img_tensor.repeat(3,1,1,1)/255
        
        central_mass_preds_h = (preds[:, 0] * img_mask.shape[-2]).int()
        central_mass_preds_w = (preds[:, 1] * img_mask.shape[-1]).int()

        central_mass_gt_h = (batch[2][:,0] * img_mask.shape[-2]).int()
        central_mass_gt_w = (batch[2][:,1] * img_mask.shape[-1]).int()
        
        for idx in range(central_mass_preds_h.shape[0]):
            # Set point of prediction
            pred_point_h = central_mass_preds_h[idx]
            pred_point_w = central_mass_preds_w[idx]
            img_mask[0, idx,
                pred_point_h-point_size:pred_point_h+point_size,
                pred_point_w-point_size:pred_point_w+point_size,
            ] = 1

            gt_point_h = central_mass_gt_h[idx]
            gt_point_w = central_mass_gt_w[idx]
            # Set point of ground truth
            img_mask[1, idx,
                gt_point_h-point_size:gt_point_h+point_size,
                gt_point_w-point_size:gt_point_w+point_size,
            ] = 1

        log_imgs = img_mask.swapaxes(1,0).cpu()
        log_img_paths = []
        for idx, img in enumerate(log_imgs):
            img_path = os.path.join(self.hparams.eval_img_path, f"{phase}-{batch_idx}-{idx}.png")
            save_image(
                img,
                img_path
            )
            log_img_paths.append(img_path)

        return log_img_paths

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

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        img_log_paths = [path for output in outputs for path in output["img_log_paths"]]
        if len(img_log_paths) > 0:
            self.logger.log_image("train/images", img_log_paths)
        
    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        img_log_paths = self.log_batch_img(batch, batch_idx, preds, targets, phase="val") if self.log_val_img else []

        return {
            "loss": loss,
            "img_log_paths": img_log_paths
        }


    def validation_epoch_end(self, outputs: List[Any]):
        img_log_paths = [path for output in outputs for path in output["img_log_paths"]]
        if len(img_log_paths) > 0:
            self.logger.log_image("val/images", img_log_paths)
        

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True)

        img_log_paths = self.log_batch_img(batch, batch_idx, preds, targets, phase="test") if self.log_test_img else []

        return {
            "loss": loss,
            "img_log_paths": img_log_paths
        }

    def test_epoch_end(self, outputs: List[Any]):
        img_log_paths = [path for output in outputs for path in output["img_log_paths"]]
        if len(img_log_paths) > 0:
            self.logger.log_image("test/images", img_log_paths)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return {
            "optimizer": self.hparams.optimizer(params=self.parameters()),
        }
