"""
Experimental
"""
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
import numpy as np

class DCRFPostProcessor():
    def __init__(self):
        pass
    
    def __call__(self, img):
        IMG_SHAPE = (img.shape[-2], img.shape[-1])

        _, labels = np.unique(img, return_inverse=True)
        # _, labels = np.unique(pred_mask, return_inverse=True)
        U = unary_from_labels(labels=labels, n_labels=2, gt_prob=.8, zero_unsure=False)
        d = dcrf.DenseCRF2D(*IMG_SHAPE, 2)  # width, height, nlabels

        # U = np.expand_dims(-np.log(pred_mask), 0)     # Get the unary in some way.
        # print(U.shape)        # -> (5, 480, 640)
        # print(U.dtype)        # -> dtype('float32')
        # U = U.reshape((1,-1)).astype(np.float32) # Needs to be flat.
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=3, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        energy = create_pairwise_bilateral(sdims=(3,3), schan=0.01, img=img, chdim=-1)
        d.addPairwiseEnergy(energy, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        
        Q = d.inference(5)
        output = np.argmax(Q, axis=0).reshape(IMG_SHAPE)
        
        return output