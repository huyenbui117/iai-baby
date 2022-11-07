import math
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
        loss, preds, targets, args = self.step(batch)

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

    def on_validation_epoch_start(self) -> None:
        return super().on_validation_epoch_start()

    def validation_step(self, batch: Any, batch_idx: int):

        loss, preds, targets, _ = self.step(batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        img_log_paths = self.log_batch_img(batch, batch_idx, preds, targets, phase="val") if self.log_val_img else []

        return {
            "loss": loss,
            "img_log_paths": img_log_paths
        }

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.step(batch)

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
    

    def sample_positive_points(self, img: torch.Tensor, rate=.1):
        """
        Return the key position points of a label binary map
        - t: input tensor (h x w)
        Return normalized (r,c) 0 <= r,c <= 1
        """
        device = img.device
        index = (img == 1).nonzero().float()
        index_r = index[:,0] / img.shape[0]
        index_c = index[:,1] / img.shape[1]
        
        normalized_index = torch.stack((index_r, index_c), dim=-1).to(device)
            
        num_samples = max(2, int(normalized_index.shape[0]*rate))
        
        if len(normalized_index) == 0:
            return []

        indices = torch.tensor([i for i in range(0, len(normalized_index), len(normalized_index)//num_samples)]).to(img.device)
        
        return torch.index_select(normalized_index, 0, indices)
        

    def measure_nt_width(self, label_tensors: torch.Tensor):
        device = label_tensors.device
        keypoints = []
        for label_tensor in label_tensors:
            anchor_points = self.sample_positive_points(label_tensor, rate=.01)
            if len(anchor_points) == 0:
                keypoints.append([])
                continue
            # Do linear regression on 3 anchor points (left, center, right)
            points = torch.stack([torch.stack((p[1],p[0])) for p in anchor_points]).to(device)

            X = torch.stack((torch.ones(points.shape[0]).to(device),points[:,0]*label_tensor.shape[-1]))
            Y = points[:,1].unsqueeze(-1)*label_tensor.shape[-2]
            # b = (X'X)^-1 . X'Y
            b = torch.inverse(X @ X.transpose(0,1)) @ X @ Y # Formula for linear regression

            x0 = 0
            y0 = b[0] + b[1]*x0
            x1 = int(label_tensor.shape[-1])
            y1 = b[0] + b[1]*x1

            x0 = int(x0)
            x1 = int(x1)
            y0 = int(y0)
            y1 = int(y1)

            best_points = [] # the pixel coordinate of the point in the estimated line
            best_size = 0 # the number of pixel of the estimation line
            for x in range(0, int(label_tensor.shape[-1])):
                y = int(b[0] + b[1] * x)

                if y >= int(label_tensor.shape[-2]): 
                    continue
                if not label_tensor[y,x]:
                    continue

                b1 = -1 / b[1]
                b0 = y - b1*x
                points = []
                # y = b0 + b1x is now the equation perpendicular to the main axis
                anchor_x, anchor_y = x, y
                size_px = 0
                while label_tensor[anchor_y,anchor_x]:
                    size_px += 1
                    points.append((anchor_x,anchor_y))
                    anchor_y += 1
                    anchor_x = math.floor((anchor_y - b0) / b1)
                anchor_x, anchor_y = x, y

                points.reverse()

                while label_tensor[anchor_y,anchor_x]:
                    size_px += 1
                    points.append((anchor_x,anchor_y))
                    anchor_y -= 1
                    anchor_x = math.floor((anchor_y - b0) / b1)

                

                if size_px > best_size:
                    best_size = size_px
                    best_points = points
            if best_size == 0:
                keypoints.append([])
                continue
            keypoints.append([best_points[0], best_points[-1]])
            
        return torch.tensor(keypoints)



    def calculate_nt_thickness(self, keypoints: torch.Tensor):
        p1 = keypoints[:,0,:]
        p2 = keypoints[:,1,:]
        return torch.sqrt(torch.sum((p1-p2)**2, dim=-1))