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
            areas = components[:, 4]
            sorted_components_index = np.argsort(areas, axis=-1)
            if len(sorted_components_index) < 2:
                output_masks.append(torch.tensor(input_mask))
            else:
                index = sorted_components_index[-2]
                output_mask = cc_mask == index
                output_masks.append(torch.tensor(output_mask))
        
        return torch.stack(output_masks).to(device)


class ThresholdAreaProcessor(object):
    def __init__(self, threshold=400, top_k=3):
        """Remove small positive regions

        Args:
            threshold (int, optional): Minimum area of the region. Defaults to 500.
        """
        self.threshold = threshold
        self.top_k = top_k
    
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
            areas = components[:, 4]
            top_k_component_indexes = np.argsort(areas, axis=-1)[::-1][1:self.top_k+1]

            # import IPython ; IPython.embed()
            
            if len(top_k_component_indexes) < 1:
                output_masks.append(torch.tensor(input_mask))
            else:
                output_mask = torch.zeros(input_mask.shape)

                for idx, index in enumerate(top_k_component_indexes):
                    component_mask = torch.tensor(cc_mask == index)
                    if component_mask.sum() < self.threshold:
                        if idx == 0:
                            output_mask += component_mask
                        break
                    output_mask += component_mask
                # import IPython ; IPython.embed()
                
                # output_masks.append(torch.tensor(input_mask))
                output_masks.append(torch.tensor(output_mask).int())
                
        return torch.stack(output_masks).to(device)