from typing import Any
import albumentations
import numpy as np
import torch


class AlbumentationWrapper():
    def __init__(self, transform: albumentations.BasicTransform) -> None:
        """Albumentation transform wrapper for the pipeline

        Args:
            transforms (List): List of albumentation transformations
        """
        self.transform = transform

    def __str__(self):
        return super().__str__() + f"({self.transform})"

    def __call__(self, image: torch.Tensor, *args: Any, **kwds: Any) -> torch.Tensor:
        """Apply the albumentation transform on the image

        Args:
            image (torch.Tensor): The input tensor image (c x h x w)

        Returns:
            torch.Tensor: The output tensor image
        """
        image_np = image.numpy()
        image_np = np.transpose(image_np, (1,2,0))
        image_transformed = self.transform(image=image_np)["image"]
        image_transformed = np.transpose(image_transformed, (2,0,1))
        return torch.tensor(image_transformed).to(image.device)

if __name__ == "__main__":
    preprocessors = [
        AlbumentationWrapper(albumentations.PadIfNeeded(
            min_height=1200, min_width=1200
        )),
        AlbumentationWrapper(albumentations.ShiftScaleRotate(
            shift_limit=.2,scale_limit=.2,rotate_limit=30,p=0.5
        )),
        AlbumentationWrapper(albumentations.RandomCrop(
            height=320, width=320
        )),
        AlbumentationWrapper(albumentations.RandomBrightnessContrast(
            p=.5
        )),
        AlbumentationWrapper(albumentations.Normalize(
            mean=[0.485],
            std=[0.229]
        )),
    ]
    
    for preprocessor in preprocessors:
        print(preprocessor)
        img = torch.rand((1, 320, 544))
        output = preprocessor(img)

        print("Input:", img.shape)
        print("Output:", output.shape)
        print(output)
        print("===")