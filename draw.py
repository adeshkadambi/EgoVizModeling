'''Module for drawing bounding boxes on images.'''

import torch
import numpy as np # type: ignore

from PIL import Image # type: ignore
from pathlib import Path
from image import ImageContainer
from torchvision.io import read_image # type: ignore
from torchvision.utils import draw_bounding_boxes # type: ignore

def get_class_names(label_ids:np.ndarray, mapper:dict) -> list[str]:
    '''Gets class names from ImageContainer.
    
    args:
        mapper (dict): mapper in format:
        {
            label_id: {label: label, subclasses: [subclasses_ids], subclasses_names: [subclasses_names]},
            ...
        }
    '''
    return [mapper[label_id]['label'] for label_id in label_ids]

def draw_bbox(img_path:str, pred_boxes:torch.Tensor, pred_classes:list[str], target_boxes:torch.Tensor, target_classes:list[str]) -> torch.Tensor:
    '''Draws bounding boxes on an image.'''

    assert len(pred_boxes) == len(pred_classes) and len(target_boxes) == len(target_classes), "pred_boxes and pred_classes must be the same length, and target_boxes and target_classes must be the same length."

    # read image from path using torchvision
    img = read_image(img_path)

    # if len of pred_classes and target_classes is 0, return img
    if len(pred_classes) < 1 and len(target_classes) < 1:
        return img
    
    # draw pred boxes in red if len is > 0
    if len(pred_classes) >= 1:
        img = draw_bounding_boxes(img, boxes=pred_boxes, labels=pred_classes, colors=(255, 0, 0), width=4)
    
    # draw target boxes in green if len is > 0
    if len(target_classes) >= 1:
        img = draw_bounding_boxes(img, boxes=target_boxes, labels=target_classes, colors=(0, 255, 0), width=4)
    
    return img

def batch_draw_bbox_image_container(images: list[ImageContainer], data_folder:str, mapper:dict, save_path:str):
    '''Draws bounding boxes on a batch of images.'''

    # get a list of any jpg in all subdirectories of data_folder
    img_paths = [str(path) for path in Path(data_folder).rglob('*.jpg')]

    # exclude any images in shan or unidet folders
    img_paths = [path for path in img_paths if 'shan' not in path and 'unidet' not in path]

    # sort img_paths
    img_paths.sort()

    for image in images:
        # if image.name in img_paths, get image path
        for path in img_paths:
            if (image.name + '.jpg') == path.split('/')[-1]:
                img_path = path
                break

        # get pred boxes and classes
        pred_boxes = torch.from_numpy(image.unidet_boxes)
        pred_classes = get_class_names(image.unidet_classes, mapper)
        
        # get target boxes and classes
        target_boxes = torch.from_numpy(image.gt_boxes)
        target_classes = get_class_names(image.gt_classes, mapper)
        
        # draw bounding boxes on image
        out: torch.Tensor = draw_bbox(img_path, pred_boxes, pred_classes, target_boxes, target_classes)
        
        # define save directory and check if it exists
        new_path: str = save_path + img_path.split(data_folder)[1].split(image.name)[0]
        save_dir = Path(new_path)

        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        
        # define save path
        new_path += image.name + '.jpg'

        # save image using PIL
        arr_out = out.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(arr_out)
        im.save(new_path)

