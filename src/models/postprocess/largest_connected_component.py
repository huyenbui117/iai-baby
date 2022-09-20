import torch
import numpy as np

class MaxAreaProcessor(object):
    def __init__(self):
        pass
    
    def __call__(self, input_mask: torch.Tensor) -> torch.Tensor:
        """
        input_mask: torch.Tensor (h x w) - Input mask tensor. Pixel values are 0 or 1
        Return output_mask: torch.Tensor (h x w) - Output mask tensor. Pixel values are 0 or 1
        """
        input_mask = input_mask.numpy().astype(np.uint8)
        ret, cc_mask, components, centroids = cv2.connectedComponentsWithStats(input_mask)
        sorted_components_index = np.argsort(np.max(components, axis=-1))
        if len(sorted_components_index) < 2:
            return input_mask
        else:
            index = sorted_components_index[-2]
            output_mask = cc_mask == index
            return torch.tensor(output_mask)  