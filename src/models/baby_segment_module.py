from typing import Any, List, Tuple, Union, Callable
import os
import cv2
import torch
from torchmetrics import MaxMetric, F1Score, Accuracy, Precision, Recall, JaccardIndex, MeanAbsoluteError
import segmentation_models_pytorch
from torchvision.utils import save_image

from .baby_module import BabyLitModule

class BabySegmentLitModule(BabyLitModule):
    """Example of LightningModule for Baby segmentation.
    """

    def __init__(
        self,
        net: segmentation_models_pytorch.Unet=None,
        optimizer: torch.optim.Optimizer=None,
        postprocessor: Union[Callable, None] = None,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        lr_scheduler_monitor: str = None,
        loss_func: torch.nn.CrossEntropyLoss = None,
        eval_img_path: str = "./tmp",
        log_train_img: float = 0.2,
        log_val_img: float = 1.,
        log_test_img: float = 1.,
    ):
        super().__init__(
            net=net,
            postprocessor=postprocessor,
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
        self.train_acc = Accuracy(mdmc_reduce=MDMC_REDUCE, task="binary")
        self.val_acc = Accuracy(mdmc_reduce=MDMC_REDUCE, task="binary")
        self.test_acc = Accuracy(mdmc_reduce=MDMC_REDUCE, task="binary")

        self.train_precision = Precision(mdmc_reduce=MDMC_REDUCE, task="binary", ignore_index=0)
        self.val_precision = Precision(mdmc_reduce=MDMC_REDUCE, task="binary", ignore_index=0)
        self.test_precision = Precision(mdmc_reduce=MDMC_REDUCE, task="binary", ignore_index=0)

        self.train_recall = Recall(mdmc_reduce=MDMC_REDUCE, task="binary", ignore_index=0)
        self.val_recall = Recall(mdmc_reduce=MDMC_REDUCE, task="binary", ignore_index=0)
        self.test_recall = Recall(mdmc_reduce=MDMC_REDUCE, task="binary", ignore_index=0)

        self.train_f1 = F1Score(mdmc_reduce=MDMC_REDUCE, task="binary", ignore_index=0)
        self.val_f1 = F1Score(mdmc_reduce=MDMC_REDUCE, task="binary", ignore_index=0)
        self.test_f1 = F1Score(mdmc_reduce=MDMC_REDUCE, task="binary", ignore_index=0)

        self.train_iou = JaccardIndex(mdmc_reduce=MDMC_REDUCE, task="binary", num_classes=2, average=None)
        self.val_iou = JaccardIndex(mdmc_reduce=MDMC_REDUCE, task="binary", num_classes=2, average=None)
        self.test_iou = JaccardIndex(mdmc_reduce=MDMC_REDUCE, task="binary", num_classes=2, average=None)

        self.test_keypoints_mae = MeanAbsoluteError()

        # for logging best so far validation accuracy
        self.val_f1_best = MaxMetric()
        self.val_iou_best = MaxMetric()


    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_iou_best doesn't store accuracy from these checks
        self.net.train()
        self.val_f1_best.reset()
        self.val_iou_best.reset()


    def step(self, batch: Any):
        x, y, args = batch
        assert x.shape == y.shape
        assert len(y.unique()) == 2
        logits = self.forward(x)

        preds = torch.argmax(logits, dim=1)
        y = y.squeeze(1).long()
        loss = self.criterion(logits, y)

        if self.postprocessor is not None:
            preds = self.postprocessor(preds)

        return loss, preds, y, args

    
    def plot_keypoints(self, 
        keypoints: torch.Tensor, 
        img_mask: torch.Tensor,
        viz_point_size: int = 5,
        color_code: Tuple[int,int,int] = (0,0,1)
    ):
        width, height = img_mask.shape[-1], img_mask.shape[-2]

        for idx, pred_keypoints in enumerate(keypoints):
            for x, y in pred_keypoints:
                for channel in range(len(color_code)):
                    img_mask[
                        channel,
                        idx,
                        max(0,y-viz_point_size)
                            :min(height-1,y+viz_point_size),
                        max(0,x-viz_point_size)
                            :min(width-1,x+viz_point_size)] = color_code[channel]
        return img_mask

    
    def plot_thickness(
        self,
        images: torch.Tensor, # shape n x c x h x w
        pred_thickness=None,
        gt_thickness=None,
        pred_keypoints=None,
        gt_keypoints=None,
        pred_color=(1,0,0),
        gt_color=(0,1,0),
        font = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 1,
        thickness = 2,
    ):
        new_images = []
        for idx, image in enumerate(images):
            image = image.cpu().numpy().transpose((1,2,0)).copy()
            w = image.shape[1]
            cv2.putText(image, f"PR: {pred_thickness[idx]:.2f}", (w-200,150), font, 
                            fontScale, pred_color, thickness, cv2.LINE_AA)
            cv2.putText(image, f"GT: {gt_thickness[idx]:.2f}", (w-200,200), font, 
                            fontScale, gt_color, thickness, cv2.LINE_AA)
            
            if pred_keypoints.shape[-1] > 0:
                cv2.line(
                    image,
                    pred_keypoints[idx][0].tolist(), 
                    pred_keypoints[idx][1].tolist(), 
                    pred_color, 
                    thickness
                )
            cv2.line(
                image,
                gt_keypoints[idx][0].tolist(), 
                gt_keypoints[idx][1].tolist(), 
                gt_color, 
                thickness
            )
            
            new_images.append(torch.from_numpy(image.transpose(2,0,1)))
        new_images = torch.stack(new_images, dim=0)

        return new_images.to(images.device)


    def log_batch_img(
        self, 
        batch, 
        batch_idx, 
        preds, 
        targets, 
        phase="train", 
        **kwargs
    ):
        # Calculate and log predicted images
        img_tensor = batch[0].swapaxes(0,1) # img_tensor shape: (channel(1) x n x h x w)
        img_mask = img_tensor.repeat(3,1,1,1)

        # Mask the prediction with red, groundtruth with green 
        img_mask[0] = (preds == 1) * 0.1 + (img_tensor[0] != 1) * img_mask[1]
        img_mask[1] = (targets == 1) * 0.1 + (img_tensor[0] != 1) * img_mask[1]

        # Draw the keypoints if they are provided
        # if "pred_keypoints" in kwargs:
        #     img_mask = self.plot_keypoints(kwargs["pred_keypoints"], img_mask, color_code=(0,0,1))

        # if "gt_keypoints" in kwargs:
        #     img_mask = self.plot_keypoints(kwargs["gt_keypoints"], img_mask, color_code=(0,1,0))
        
        y_mean = torch.mean(preds.nonzero()[:,1].float()).int()
        x_mean = torch.mean(preds.nonzero()[:,2].float()).int()
        x,y = kwargs.get("gt_keypoints")[0][0]
        if x_mean < img_mask.shape[-1]/2: # region is on the left
            if x > img_mask.shape[-1]/2:
                # flip right->left
                x1 = kwargs["gt_keypoints"][0][0][0]
                x2 = kwargs["gt_keypoints"][0][1][0]
                kwargs["gt_keypoints"][0][0][0] = x1 - (x1 - img_mask.shape[-1]//2)*2
                kwargs["gt_keypoints"][0][1][0] = x2 - (x2 - img_mask.shape[-1]//2)*2
            # if y < img_mask.shape[-2]/2:
            #     # flip top->bottom
            #     y1 = kwargs["gt_keypoints"][0][0][1]
            #     y2 = kwargs["gt_keypoints"][0][1][1]
            #     kwargs["gt_keypoints"][0][0][1] = y1 - (y1 - img_mask.shape[-2]//2)*2
            #     kwargs["gt_keypoints"][0][1][1] = y2 - (y2 - img_mask.shape[-2]//2)*2
        else: # region is on the right
            if x < img_mask.shape[-1]/2:
                # flip right->left
                x1 = kwargs["gt_keypoints"][0][0][0]
                x2 = kwargs["gt_keypoints"][0][1][0]
                kwargs["gt_keypoints"][0][0][0] = x1 - (x1 - img_mask.shape[-1]//2)*2
                kwargs["gt_keypoints"][0][1][0] = x2 - (x2 - img_mask.shape[-1]//2)*2
            # if y < img_mask.shape[-2]/2:
            #     # fip top->bottom
            #     y1 = kwargs["gt_keypoints"][0][0][1]
            #     y2 = kwargs["gt_keypoints"][0][1][1]
            #     kwargs["gt_keypoints"][0][0][1] = y1 - (y1 - img_mask.shape[-2]//2)*2
            #     kwargs["gt_keypoints"][0][1][1] = y2 - (y2 - img_mask.shape[-2]//2)*2
        # import IPython ; IPython.embed()

        log_imgs = img_mask.swapaxes(1,0).cpu()
        log_imgs = self.plot_thickness(
            log_imgs,
            pred_thickness=kwargs.get("pred_nt_thickness"),    
            gt_thickness=kwargs.get("gt_nt_thickness"),
            pred_keypoints=kwargs.get("pred_keypoints"),
            gt_keypoints=kwargs.get("gt_keypoints")
        )

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
        if not self.net.training:
            self.net.train()
        loss, preds, targets, _ = self.step(batch)

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

        img_log_paths = self.log_batch_img(batch, batch_idx, preds, targets, phase="train") if self.log_train_img > 0 else []

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {
            "loss": loss,
            "img_log_paths": img_log_paths
        }


    def training_epoch_end(self, outputs: List[Any]):
        self.log_images("train/images", outputs, log_ratio=self.log_train_img)
        self.train_acc.reset()
        return super().training_epoch_end(outputs)

    def on_validation_epoch_start(self) -> None:
        self.net.eval()
        return super().on_validation_epoch_start()


    def validation_step(self, batch: Any, batch_idx: int):
        if self.net.training:
            self.net.eval()
        loss, preds, targets, _ = self.step(batch)

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

        img_log_paths = self.log_batch_img(batch, batch_idx, preds, targets, phase="val") if self.log_val_img > 0 else []
        
        return {
            "loss": loss,
            "img_log_paths": img_log_paths
        }


    def validation_epoch_end(self, outputs: List[Any]):
        iou = self.val_iou.compute()  # get val iou from current epoch
        f1 = self.val_f1.compute()  # get val f1 from current epoch

        self.val_f1_best.update(f1)
        self.val_iou_best.update(iou[1])
        
        self.log("val/iou_best", self.val_iou_best.compute(), on_epoch=True, prog_bar=True)
        self.log("val/f1_best", self.val_f1_best.compute(), on_epoch=True, prog_bar=True)
        self.log_images("val/images", outputs, log_ratio=self.log_val_img)
        
        self.val_iou.reset()
        self.val_f1.reset()

        return super().validation_epoch_end(outputs)


    def test_step(self, batch: Any, batch_idx: int):
        if self.net.training:
            self.net.eval()
        loss, preds, targets, args = self.step(batch)

        if "pred_keypoints" in args:
            pred_keypoints = args["pred_keypoints"]

        if "pred_keypoints" in args:
            gt_keypoints = args["gt_keypoints"]

        # log test metrics
        acc = self.test_acc(preds, targets)
        precision = self.test_precision(preds, targets)
        recall = self.test_recall(preds, targets)
        f1 = self.test_f1(preds, targets)
        iou = self.test_iou(preds, targets)[1]

        gt_nt_thickness = self.calculate_nt_thickness(gt_keypoints).to(self.device)
        if pred_keypoints.shape[-1] != 0:        
            pred_nt_thickness = self.calculate_nt_thickness(pred_keypoints).to(self.device)
            keypoints_err = torch.mean(
                torch.sqrt((gt_nt_thickness-pred_nt_thickness)**2)
            )
            self.log("test/nt_keypoints_error/exclude_undetected", keypoints_err, on_step=False, on_epoch=True)
        else:
            pred_nt_thickness = torch.zeros(gt_keypoints.shape[0]).to(self.device)
            keypoints_err = torch.mean(
                torch.sqrt((gt_nt_thickness-pred_nt_thickness)**2)
            )

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/precision", precision, on_step=False, on_epoch=True)
        self.log("test/recall", recall, on_step=False, on_epoch=True)
        self.log("test/f1", f1, on_step=False, on_epoch=True)
        self.log("test/iou", iou, on_step=False, on_epoch=True)
        self.log("test/nt_keypoints_error", keypoints_err, on_step=False, on_epoch=True)
        self.log("test/nt_size", gt_nt_thickness, on_step=False, on_epoch=True)

        img_log_paths = self.log_batch_img(
            batch, 
            batch_idx, 
            preds, 
            targets, 
            phase="test",
            pred_keypoints=pred_keypoints,
            gt_keypoints=gt_keypoints,
            pred_nt_thickness=pred_nt_thickness,
            gt_nt_thickness=gt_nt_thickness
        ) if self.log_val_img > 0 else []

        return {
            "loss": loss,
            "img_log_paths": img_log_paths
        }


    def test_epoch_end(self, outputs: List[Any]):
        self.log_images("test/images", outputs, log_ratio=self.log_test_img)

        self.test_acc.reset()

        return super().test_epoch_end(outputs)


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
