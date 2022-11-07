import os
from typing import Any, List, Union, Callable
import torch
from .baby_segment_module import BabySegmentLitModule
from torchvision.utils import save_image

class BabySegmentWithHeadMaskLitModule(BabySegmentLitModule):
    """LitModule for NT Segmentation with auxilary input head mask
    """

    def __init__(
        self,
        head_output_auxilary=False,
        **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self.head_output_auxilary = head_output_auxilary

    def step(self, batch: Any):
        x, y, y_head = batch
        
        if self.head_output_auxilary:
            pass
        else:
            logits = self.forward(torch.cat((x, y_head), dim=1))
        preds = torch.argmax(logits, dim=1)
        y = y.squeeze(1).long()
        loss = self.criterion(logits, y)

        if self.postprocessor is not None:
            preds = self.postprocessor(preds)

        return loss, preds, y

    
    def log_batch_img(self, batch, batch_idx, preds, targets, phase="train"):
        # Calculate and log predicted images
        img_tensor = batch[0].swapaxes(0,1) # img_tensor shape: (channel(1) x n x h x w)
        img_mask = img_tensor.repeat(3,1,1,1)

        head_label_tensor = batch[2].swapaxes(0,1)
        head_label_mask = head_label_tensor.repeat(3,1,1,1)

        # Mask the prediction with red, groundtruth with green 
        img_mask[0] = (preds == 1) * 1 + (img_tensor[0] != 1) * img_mask[1]
        img_mask[1] = (targets == 1) * 1 + (img_tensor[0] != 1) * img_mask[1]
            
        log_imgs = torch.cat((img_mask, head_label_mask), dim=2).swapaxes(1,0).cpu()
        log_img_paths = []
        for idx, img in enumerate(log_imgs):
            img_path = os.path.join(self.hparams.eval_img_path, f"{phase}-{batch_idx}-{idx}.png")
            save_image(
                img,
                img_path
            )
            log_img_paths.append(img_path)

        return log_img_paths