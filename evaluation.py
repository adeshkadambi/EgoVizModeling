'''Module for calculation of evaluation metrics for model performance.'''

import torch # type: ignore

from torchmetrics.detection.mean_ap import MeanAveragePrecision # type: ignore
from torchvision import ops # type: ignore


def single_img_mean_iou():
    '''Calculate mean IoU for a single image.'''
    
    