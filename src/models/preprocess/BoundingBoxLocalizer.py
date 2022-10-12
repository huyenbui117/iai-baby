from typing import Any
import albumentations
import numpy as np
import torch


def calculate_bboxes(label_masks: torch.Tensor, loosen_amount: float = 0.03):
    """
    Return the bounding boxes of tensor
    - label_masks: input tensor List(1 x h x w)
    Return:
    - bboxes: List[Tuple(float, float, float, float)] - List of bounding boxes (center_r, center_c, w, h) for all batches in the tensor
    """
    bboxes = []
    for t in label_masks:
        index = (t == 1).nonzero()
            
        rmax_id = torch.argmax(index[:,-2])
        cmax_id = torch.argmax(index[:,-1])
        rmin_id = torch.argmin(index[:,-2])
        cmin_id = torch.argmin(index[:,-1])
        
        r0, c0 = index[rmin_id,1], index[cmin_id, 2]
        r1, c1 = index[rmax_id,1], index[cmax_id, 2]

        r0 = max(r0 / t.shape[-2] - loosen_amount, torch.tensor(0))
        r1 = min(r1 / t.shape[-2] + loosen_amount, torch.tensor(1))
        c0 = max(c0 / t.shape[-1] - loosen_amount, torch.tensor(0))
        c1 = min(c1 / t.shape[-1] + loosen_amount, torch.tensor(1))
        bboxes.append(torch.stack((r0, c0, r1, c1)))

    return bboxes


class SimpleLocalizer():
    def __init__(self) -> None:
        """A simple localizer that is based on the segmentation mask
        """
        pass

    def __str__(self):
        return super().__str__() + f"({self.transform})"

    def __call__(self, image: torch.Tensor, label: torch.Tensor, *args: Any, **kwds: Any) -> torch.Tensor:
        """Apply the localizer on the image

        Args:
            image (torch.Tensor): The input tensor image and label (c x h x w)
            label (torch.Tensor): The input tensor mask and label (c x h x w)

        Returns:
            torch.Tensor: The output tensor image, including 
        """
        
        bboxes = calculate_bboxes([label], loosen_amount=0.05)
        r0, c0, r1, c1 = bboxes[0]

        r0 = int(r0 * img.shape[-2])
        r1 = int(r1 * img.shape[-2])
        c0 = int(c0 * img.shape[-1])
        c1 = int(c1 * img.shape[-1])

        img = img[:, r0:r1+1, c0:c1+1]
        label = label[:, r0:r1+1, c0:c1+1]


if __name__ == "__main__":
    preprocessors = [
    ]
    
    for preprocessor in preprocessors:
        print(preprocessor)
        img = torch.rand((1, 320, 544))
        output = preprocessor(img)

        print("Input:", img.shape)
        print("Output:", output.shape)
        print(output)
        print("===")