from typing import Any, List
import os
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, F1Score, Accuracy, Precision, Recall, JaccardIndex
import segmentation_models_pytorch
from torchvision.utils import save_image

from .baby_module import BabyLitModule

class BabySegmentLitModule(BabyLitModule):
    """Example of LightningModule for Baby segmentation.
    """

    def __init__(
        self,
        net: segmentation_models_pytorch.Unet,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        lr_scheduler_monitor: str = None,
        loss_func: torch.nn.CrossEntropyLoss = None,
        eval_img_path: str = "./tmp",
        log_train_img: bool = True,
        log_val_img: bool = True,
        log_test_img: bool = True,
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

        MDMC_REDUCE = "samplewise"

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy(mdmc_reduce=MDMC_REDUCE)
        self.val_acc = Accuracy(mdmc_reduce=MDMC_REDUCE)
        self.test_acc = Accuracy(mdmc_reduce=MDMC_REDUCE)

        self.train_precision = Precision(mdmc_reduce=MDMC_REDUCE, ignore_index=0)
        self.val_precision = Precision(mdmc_reduce=MDMC_REDUCE, ignore_index=0)
        self.test_precision = Precision(mdmc_reduce=MDMC_REDUCE, ignore_index=0)

        self.train_recall = Recall(mdmc_reduce=MDMC_REDUCE, ignore_index=0)
        self.val_recall = Recall(mdmc_reduce=MDMC_REDUCE, ignore_index=0)
        self.test_recall = Recall(mdmc_reduce=MDMC_REDUCE, ignore_index=0)

        self.train_f1 = F1Score(mdmc_reduce=MDMC_REDUCE, ignore_index=0)
        self.val_f1 = F1Score(mdmc_reduce=MDMC_REDUCE, ignore_index=0)
        self.test_f1 = F1Score(mdmc_reduce=MDMC_REDUCE, ignore_index=0)

        self.train_iou = JaccardIndex(mdmc_reduce=MDMC_REDUCE, num_classes=2, average=None)
        self.val_iou = JaccardIndex(mdmc_reduce=MDMC_REDUCE, num_classes=2, average=None)
        self.test_iou = JaccardIndex(mdmc_reduce=MDMC_REDUCE, num_classes=2, average=None)

        # for logging best so far validation accuracy
        self.val_f1_best = MaxMetric()
        self.val_iou_best = MaxMetric()


    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_iou_best doesn't store accuracy from these checks
        self.val_f1_best.reset()
        self.val_iou_best.reset()


    def step(self, batch: Any):
        x, y = batch

        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        y = y.squeeze(1).long()
        loss = self.criterion(logits, y)
        return loss, preds, y


    def log_batch_img(self, batch, batch_idx, preds, targets, phase="train"):
        # Calculate and log predicted images
        img_tensor = batch[0].swapaxes(0,1) # img_tensor shape: (channel(1) x n x h x w)
        img_mask = img_tensor.repeat(3,1,1,1)

        # Mask the prediction with red, groundtruth with green 
        img_mask[0] = (preds == 1) * 1 + (img_tensor[0] != 1) * img_mask[1]
        img_mask[1] = (targets == 1) * 1 + (img_tensor[0] != 1) * img_mask[1]
            
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

        # log train metrics
        # preds shape   :   (batch_size x h x w)
        # targets shape :   (batch_size x h x w)
        acc = self.train_acc(preds, targets) 
        precision = self.train_precision(preds, targets) 
        recall = self.train_recall(preds, targets) 
        f1 = self.train_f1(preds, targets) 
        iou = self.train_iou(preds, targets)[1]

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision", precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recall", recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/iou", iou, on_step=False, on_epoch=True, prog_bar=True)

        img_log_paths = self.log_batch_img(batch, batch_idx, preds, targets, phase="train") if self.log_train_img else []

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {
            "loss": loss,
            "img_log_paths": img_log_paths
        }


    def training_epoch_end(self, outputs: List[Any]):
        self.log_images("train/images", outputs)
        self.train_acc.reset()


    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        precision = self.val_precision(preds, targets)
        recall = self.val_recall(preds, targets)
        f1 = self.val_f1(preds, targets)
        iou = self.val_iou(preds, targets)[1]
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou", iou, on_step=False, on_epoch=True, prog_bar=True)

        img_log_paths = self.log_batch_img(batch, batch_idx, preds, targets, phase="val") if self.log_val_img else []
        
        return {
            "loss": loss,
            "img_log_paths": img_log_paths
        }


    def validation_epoch_end(self, outputs: List[Any]):
        iou = self.val_iou.compute()  # get val iou from current epoch
        f1 = self.val_f1.compute()  # get val f1 from current epoch
        
        self.log("val/iou_best", iou[1], on_epoch=True, prog_bar=True)
        self.log("val/f1_best", f1, on_epoch=True, prog_bar=True)
        self.log_images("val/images", outputs)
        
        self.val_iou.reset()
        self.val_f1.reset()


    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        precision = self.test_precision(preds, targets)
        recall = self.test_recall(preds, targets)
        f1 = self.test_f1(preds, targets)
        iou = self.test_iou(preds, targets)[1]

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/precision", precision, on_step=False, on_epoch=True)
        self.log("test/recall", recall, on_step=False, on_epoch=True)
        self.log("test/f1", f1, on_step=False, on_epoch=True)
        self.log("test/iou", iou, on_step=False, on_epoch=True)

        img_log_paths = self.log_batch_img(batch, batch_idx, preds, targets, phase="test") if self.log_val_img else []

        return {
            "loss": loss,
            "img_log_paths": img_log_paths
        }


    def test_epoch_end(self, outputs: List[Any]):
        self.log_images("test/images", outputs)

        self.test_acc.reset()


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
