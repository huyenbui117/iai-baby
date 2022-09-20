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
        detector: Callable,
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
        self.detector = detector
    

    def localize_batch(self, x, y, batch_bboxes):
        release_pixel = 5
        localized_x, localized_y = [], []
        downsize = transforms.Resize((320, 544))
        for idx, sample in enumerate(x):
            bboxes = batch_bboxes[idx]
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
        x, y, *args = batch
        original_size = (x.shape[-2], x.shape[-1])

        if len(args) > 0:
            batch_bboxes = args[0]

            localized_x, localized_y = self.localize_batch(
                x, y, batch_bboxes
            )

        # logits = self.forward(x)
        logits = self.forward(localized_x)
        preds = torch.argmax(logits, dim=1)
        
        localized_y = localized_y.squeeze(1).long()
        loss = self.criterion(logits, localized_y)

        return loss, preds, localized_y

        # upsize = transforms.Resize(original_size)
        # logits = upsize(logits)

        # import IPython ; IPython.embed()

        # y = y.squeeze(1).long()
        # loss = self.criterion(logits, y)

        # if self.postprocessor is not None:
        #     preds = self.postprocessor(preds)

        # return loss, preds, y


    def log_batch_img(self, batch, batch_idx, preds, targets, phase="train"):
        x, y, *args = batch

        if len(args) > 0:
            batch_bboxes = args[0]

            localized_x, localized_y = self.localize_batch(
                x, y, batch_bboxes
            )
        # Calculate and log predicted images
        img_tensor = localized_x.swapaxes(0,1) # img_tensor shape: (channel(1) x n x h x w)
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



if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)