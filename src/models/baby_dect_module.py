from typing import Any, List
import os
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, F1Score, Accuracy, Precision, Recall, JaccardIndex
import segmentation_models_pytorch
from torchvision.utils import save_image

from .baby_module import BabyLitModule

class BabyDectLitModule(BabyLitModule):
    """Example of LightningModule for Baby point detection.
    """

    def __init__(
        self,
        net: segmentation_models_pytorch.Unet=None,
        optimizer: torch.optim.Optimizer=None,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        lr_scheduler_monitor: str = None,
        loss_func: torch.nn.CrossEntropyLoss=None,
        eval_img_path: str = "./tmp",
        log_train_img: float = 0.2,
        log_val_img: float = 1.,
        log_test_img: float = 1.,
    ):
        super().__init__(
            net=net,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_monitor=lr_scheduler_monitor,
            loss_func=loss_func,
            eval_img_path=eval_img_path,
            log_train_img=log_train_img,
            log_val_img=log_val_img,
            log_test_img=log_test_img
        )


    def step(self, batch: Any):
        x, _, y = batch
        
        preds = self.forward(x.float())

        # y ~ normalized coordinate (x1,y1)
        loss = self.criterion(preds, y)
        return loss, preds, y

    
    def training_epoch_end(self, outputs: List[Any]):
        self.log_images("train/images", outputs)

    
    def validation_epoch_end(self, outputs: List[Any]):
        self.log_images("val/images", outputs)

    
    def test_epoch_end(self, outputs: List[Any]):
        self.log_images("test/images", outputs)


    def log_batch_img(self, batch, batch_idx, preds, targets, phase="train", point_size=7):
        # Calculate and log predicted images
        img_tensor = batch[0].swapaxes(0,1) # img_tensor shape: (channel(1) x n x h x w)
        img_mask = img_tensor.repeat(3,1,1,1)
        
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