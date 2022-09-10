import torch

class LargestConnectedComponent:
    def __call__(self, img_tensor: torch.Tensor):
        """
        Return img tensor with the largest connected component
        - img_tensor: mask with 1/0 value (h x w)
        Return: tensor with all small connected components removed
        """

        area_mask = torch.zeros((*img_tensor.shape))

        def calculate_area_mask(i, j):
            pass
        
        def spread_area_mask(i, j):
            pass


        for i, row in enumerate(img_tensor):
            for j, cell in enumerate(row):
                if cell == 1 and area_mask[i,j] != 0:
                    area_mask[i,j] = calculate_area_mask(i, j)
                    spread_area_mask(i,j)

        indexes = torch.argmax(area_mask)
        return 

    def __repr__(self):
        return self.__class__.__name__ + '()'