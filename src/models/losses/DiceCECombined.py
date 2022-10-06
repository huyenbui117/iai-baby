import torch
from torch.nn import CrossEntropyLoss
from segmentation_models_pytorch.losses import DiceLoss

class DiceCECombined(torch.nn.Module):
    def __init__(self, 
        ce_weight=0.8, 
        dice_weight=0.2, 
        ce_loss_weight=torch.tensor([0.01, 0.99]),
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.ce = CrossEntropyLoss(
            weight=ce_loss_weight,
            label_smoothing=0.01
        )
        self.dice = DiceLoss(
            mode="multiclass",
            classes=2,
            log_loss=False,
            from_logits=True,
            smooth=0.01
        )

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate loss

        Args:
            preds (torch.Tensor): prediction tensor
            targets (torch.Tensor): ground truth tensor

        Returns:
            torch.Tensor: loss value
        """
        ce_loss = self.ce(preds, targets)
        dice_loss = self.dice(preds, targets)
        
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return loss