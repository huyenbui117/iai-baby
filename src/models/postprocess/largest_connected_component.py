import cv2
import torch
import numpy as np

class MaxAreaProcessor(object):
    def __init__(self):
        pass
    
    def __call__(self, input_masks: torch.Tensor) -> torch.Tensor:
        """
        input_masks: torch.Tensor (b x h x w) - Input mask tensor. Pixel values are 0 or 1
        Return output_mask: torch.Tensor (b x h x w) - Output mask tensor. Pixel values are 0 or 1
        """
        device = input_masks.device
        input_masks = input_masks.cpu().numpy().astype(np.uint8)
        output_masks = []
        
        for input_mask in input_masks:
            ret, cc_mask, components, centroids = cv2.connectedComponentsWithStats(input_mask)
            sorted_components_index = np.argsort(np.max(components, axis=-1))
            if len(sorted_components_index) < 2:
                output_masks.append(torch.tensor(input_mask))
            else:
                index = sorted_components_index[-2]
                output_mask = cc_mask == index
                output_masks.append(torch.tensor(output_mask))
        
        return torch.stack(output_masks).to(device)