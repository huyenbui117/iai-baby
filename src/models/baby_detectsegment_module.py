from typing import Any, List, Union, Callable
import os
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, F1Score, Accuracy, Precision, Recall, JaccardIndex
import segmentation_models_pytorch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from .baby_segment_module import BabySegmentLitModule

class BabyDetectSegmentLitModule(BabySegmentLitModule):
    """Example of LightningModule for Baby segmentation.
    """

    def __init__(
        self,
        net: segmentation_models_pytorch.Unet,
        optimizer: torch.optim.Optimizer,
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
    

    def localize_batch(self, x, y, batch_bboxes):
        release_pixel = 0
        localized_x, localized_y = [], []
        downsize = transforms.Resize((320, 544))
        for idx, sample in enumerate(x):
            bboxes = batch_bboxes[idx]["bbox"][0]
            bx, by, bw, bh = list(map(int, bboxes))
            bx0 = max(0, bx - release_pixel)
            by0 = max(0, by - release_pixel)
            bw += release_pixel*2
            bh += release_pixel*2
            bx1 = min(bx0+bw, sample.shape[-1])
            by1 = min(by0+bh, sample.shape[-2])

            localized_sample = sample[:, by0:by1, bx0:bx1]
            localized_label = y[idx, :, by0:by1, bx0:bx1]
            localized_x.append(downsize(localized_sample))
            localized_y.append(downsize(localized_label))
        # y = downsize(y)
        localized_x = torch.stack(localized_x)
        localized_y = torch.stack(localized_y)
        
        return localized_x, localized_y

    def step(self, batch: Any):
        x, y, args = batch

        if "pred_boxes" in args:
            batch_bboxes = args["pred_boxes"]

            localized_x, localized_y = self.localize_batch(
                x, y, batch_bboxes
            )


        if "gt_keypoints" in args:
            gt_keypoints = args["gt_keypoints"]
        else:
            gt_keypoints = []

        # logits = self.forward(x)
        logits = self.forward(localized_x)
        localized_preds = torch.argmax(logits, dim=1)
        
        localized_y = localized_y.squeeze(1).long()
        loss = self.criterion(logits, localized_y)

        # Calculate and log original predicted images
        original_img_mask = x
        original_img_mask = (original_img_mask*0).long()

        for idx in range(len(batch_bboxes)):
            bbox = batch_bboxes[idx]["bbox"][0]
            bx, by, bw, bh = list(map(int, bbox))
            bh = min(bh, original_img_mask.shape[-2]-by)
            bw = min(bw, original_img_mask.shape[-1]-bx)
            reconstructed_img_mask = transforms.Resize((bh, bw))(localized_preds.unsqueeze(1)[idx])
            
            original_img_mask[idx,:,by:by+bh,bx:bx+bw] = reconstructed_img_mask

        y = y.squeeze(0).long()
        original_img_mask = original_img_mask.squeeze(1).long()
        if self.postprocessor is not None:
            original_img_mask = self.postprocessor(original_img_mask)
        
        pred_keypoints = self.measure_nt_width(original_img_mask)
        return loss, original_img_mask, y, {
            "pred_keypoints": pred_keypoints,
            "gt_keypoints": gt_keypoints
        } 


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
