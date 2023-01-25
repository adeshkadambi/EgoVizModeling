"""Module for calculation of evaluation metrics for model performance."""

import torch  # type: ignore
import numpy as np
from torchvision import ops  # type: ignore
from torchmetrics.detection.mean_ap import MeanAveragePrecision  # type: ignore

from image import ImageContainer

# Localization Metrics
# --------------------------------------------------
def single_img_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor):
    """Calculate IoU for a single image.

    Args:
        pred_boxes (torch.Tensor): Predicted bounding boxes of shape (num_boxes, 4).
        target_boxes (torch.Tensor): Target bounding boxes of shape (num_boxes, 4).

    Returns:
        float: IoU for the image.
    """
    if len(pred_boxes) == 0 and len(target_boxes) == 0:
        return 1.0
    if len(pred_boxes) == 0 or len(target_boxes) == 0:
        return 0.0
    ious = ops.box_iou(pred_boxes, target_boxes)

    return ious.max(dim=1)[0].numpy()


def single_img_mean_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor):
    """Calculate mean IoU for a single image.

    Args:
        pred_boxes (torch.Tensor): Predicted bounding boxes of shape (num_boxes, 4).
        target_boxes (torch.Tensor): Target bounding boxes of shape (num_boxes, 4).

    Returns:
        float: Mean IoU for the image.
    """
    if len(pred_boxes) == 0 and len(target_boxes) == 0:
        return 1.0
    if len(pred_boxes) == 0 or len(target_boxes) == 0:
        return 0.0
    ious = ops.box_iou(pred_boxes, target_boxes)

    return ious.max(dim=1)[0].mean().item()


def batch_mean_iou(images: list[ImageContainer]):
    """Calculate mean IoU for a batch of images.

    Args:
        images (list[ImageContainer]): List of ImageContainer objects.

    Returns:
        float: Mean IoU for all objects.
        float: Mean IoU for active objects.
    """
    all_ious = np.array([])
    active_ious = np.array([])

    for img in images:
        pred = torch.from_numpy(img.unidet_boxes)
        target = torch.from_numpy(img.gt_boxes)

        active_pred = torch.from_numpy(img.unidet_active_boxes)
        active_target = torch.from_numpy(img.gt_active_boxes)

        all_ious = np.append(all_ious, single_img_iou(pred, target))
        active_ious = np.append(active_ious, single_img_iou(active_pred, active_target))

    return np.mean(all_ious), np.mean(active_ious)


def single_img_precision_recall_f1(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-10,
):
    """Calculate Precision, Recall, and F1 score for a single image.

    Args:
        pred_boxes (torch.Tensor): Predicted bounding boxes of shape (N, 4).
        target_boxes (torch.Tensor): Target bounding boxes of shape (M, 4).
        threshold (float, optional): Threshold for IoU. Defaults to 0.5.

    Returns:
        float: Precision for the image.
        float: Recall for the image.
        float: F1 score for the image.
    """
    if len(pred_boxes) == 0 and len(target_boxes) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_boxes) == 0 or len(target_boxes) == 0:
        return 0.0, 0.0, 0.0

    ious = ops.box_iou(pred_boxes, target_boxes)

    tp = (ious.max(dim=1)[0] > threshold).sum().item()
    fp = len(pred_boxes) - tp
    fn = len(target_boxes) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    return precision, recall, f1


def batch_precision_recall_f1(images: list[ImageContainer]):
    """Calculate Precision, Recall, and F1 score for a batch of images."""

    all_precisions = np.array([])
    all_recalls = np.array([])
    all_f1s = np.array([])

    active_precisions = np.array([])
    active_recalls = np.array([])
    active_f1s = np.array([])

    for img in images:
        pred = torch.from_numpy(img.unidet_boxes)
        target = torch.from_numpy(img.gt_boxes)

        active_pred = torch.from_numpy(img.unidet_active_boxes)
        active_target = torch.from_numpy(img.gt_active_boxes)

        precision, recall, f1 = single_img_precision_recall_f1(pred, target)
        all_precisions = np.append(all_precisions, precision)
        all_recalls = np.append(all_recalls, recall)
        all_f1s = np.append(all_f1s, f1)

        precision, recall, f1 = single_img_precision_recall_f1(
            active_pred, active_target
        )
        active_precisions = np.append(active_precisions, precision)
        active_recalls = np.append(active_recalls, recall)
        active_f1s = np.append(active_f1s, f1)

    return (
        np.mean(all_precisions),
        np.mean(all_recalls),
        np.mean(all_f1s),
        np.mean(active_precisions),
        np.mean(active_recalls),
        np.mean(active_f1s),
    )


# Detection Metrics
# --------------------------------------------------


def single_img_per_class_precision_recall_f1(
    pred_boxes: torch.Tensor,
    pred_classes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-10,
):
    """Calculate Precision, Recall, and F1 score for a single image per class."""

    if len(pred_boxes) == 0 and len(target_boxes) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_boxes) == 0 or len(target_boxes) == 0:
        return 0.0, 0.0, 0.0

    ious = ops.box_iou(pred_boxes, target_boxes)

    # Get unique classes
    classes = np.unique(np.concatenate((pred_classes, target_classes)))

    # Initialize arrays
    precisions = np.zeros(len(classes))
    recalls = np.zeros(len(classes))
    f1s = np.zeros(len(classes))

    for i, c in enumerate(classes):
        # Get indices of predictions and targets that belong to class c
        pred_idx = np.where(pred_classes == c)[0]
        target_idx = np.where(target_classes == c)[0]

        # Calculate TP, FP, and FN
        tp = (ious[pred_idx].max(dim=1)[0] > threshold).sum().item()
        fp = len(pred_idx) - tp
        fn = len(target_idx) - tp

        # Calculate precision, recall, and f1
        pr = tp / (tp + fp + eps)
        re = tp / (tp + fn + eps)
        f1 = (2 * pr * re) / (pr + re + eps)

        # Assign precision, recall, and f1
        precisions[i] = pr if pr <= 1 else np.nan
        recalls[i] = re if re <= 1 else np.nan
        f1s[i] = f1 if f1 <= 1 else np.nan

    # Average over classes
    precisions = np.nanmean(precisions)
    recalls = np.nanmean(recalls)
    f1s = np.nanmean(f1s)

    return precisions, recalls, f1s


def batch_macro_precision_recall_f1(images: list[ImageContainer]):
    """Calculate Precision, Recall, and F1 score for a batch of images per class."""

    pred_boxes, pred_classes, target_boxes, target_classes = (
        torch.Tensor(),
        torch.Tensor(),
        torch.Tensor(),
        torch.Tensor(),
    )
    (
        active_pred_boxes,
        active_pred_classes,
        active_target_boxes,
        active_target_classes,
    ) = (torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor())

    for img in images:
        pred_box = torch.from_numpy(img.unidet_boxes)
        pred_class = torch.from_numpy(img.unidet_classes)
        target_box = torch.from_numpy(img.gt_boxes)
        target_class = torch.from_numpy(img.gt_classes)

        active_pred_box = torch.from_numpy(img.unidet_active_boxes)
        active_pred_class = torch.from_numpy(img.unidet_active_classes)
        active_target_box = torch.from_numpy(img.gt_active_boxes)
        active_target_class = torch.from_numpy(img.gt_active_classes)

        pred_boxes = torch.cat((pred_boxes, pred_box))
        pred_classes = torch.cat((pred_classes, pred_class))
        target_boxes = torch.cat((target_boxes, target_box))
        target_classes = torch.cat((target_classes, target_class))

        active_pred_boxes = torch.cat((active_pred_boxes, active_pred_box))
        active_pred_classes = torch.cat((active_pred_classes, active_pred_class))
        active_target_boxes = torch.cat((active_target_boxes, active_target_box))
        active_target_classes = torch.cat((active_target_classes, active_target_class))

    # Calculate precision, recall, and f1 for all predictions
    precision, recall, f1 = single_img_per_class_precision_recall_f1(
        pred_boxes, pred_classes, target_boxes, target_classes
    )

    # Calculate precision, recall, and f1 for active predictions
    (
        active_precision,
        active_recall,
        active_f1,
    ) = single_img_per_class_precision_recall_f1(
        active_pred_boxes,
        active_pred_classes,
        active_target_boxes,
        active_target_classes,
    )

    return precision, recall, f1, active_precision, active_recall, active_f1


def batch_mean_ap(images: list[ImageContainer], class_metrics: bool = False):
    """Calculate mean average precision for a batch of images.

    Args:
        images (list[ImageContainer]): List of ImageContainer objects.

    Returns:
        float: Mean average precision for the batch.
    """
    preds, targets = [], []
    active_preds, active_targets = [], []

    for img in images:
        if len(img.gt_boxes) > 0:
            preds.append(
                dict(
                    boxes=torch.from_numpy(img.unidet_boxes),
                    labels=torch.from_numpy(img.unidet_classes),
                    scores=torch.from_numpy(img.unidet_scores),
                )
            )

            targets.append(
                dict(
                    boxes=torch.from_numpy(img.gt_boxes),
                    labels=torch.from_numpy(img.gt_classes),
                )
            )

        if len(img.gt_active_boxes) > 0:
            active_preds.append(
                dict(
                    boxes=torch.from_numpy(img.unidet_active_boxes),
                    labels=torch.from_numpy(img.unidet_active_classes),
                    scores=torch.from_numpy(img.unidet_active_scores),
                )
            )

            active_targets.append(
                dict(
                    boxes=torch.from_numpy(img.gt_active_boxes),
                    labels=torch.from_numpy(img.gt_active_classes),
                )
            )

    map = MeanAveragePrecision(class_metrics=class_metrics)
    map.update(preds, targets)

    active_map = MeanAveragePrecision(class_metrics=class_metrics)
    active_map.update(active_preds, active_targets)

    return map.compute()["map"], active_map.compute()["map"]
