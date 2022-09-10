import torchvision
from typing import Tuple
import random

class MultiRandomErasing(object):
    def __init__(self, 
        p: float = 0.5, 
        scale: Tuple[float, float] = (0.004, 0.01), 
        ratio: Tuple[float, float] = (0.3, 3.3), 
        value=0, 
        inplace=False,
        repeat: Tuple[int, int] = (5, 15),
    ):
        multiple = random.randrange(*repeat)
        self.transforms = [torchvision.transforms.RandomErasing(p, scale, ratio, value, inplace) for i in range(multiple)]
    
    def __call__(self, img):
        for transform in self.transforms:
            img = transform(img)
        return img