"""Object that stores the file path of the image, the ground truth labels, and shan/unidet predictions."""

import warnings
import numpy as np
import torch
import torchvision


class ImageContainer:
    """
    Object that stores the file path of the image, the ground truth labels, and shan/unidet predictions.
    """

    def __init__(self) -> None:
        self.name = None

        # targets
        self.gt_boxes = np.array([])
        self.gt_classes = np.array([])
        self.gt_active_boxes = np.array([])
        self.gt_active_classes = np.array([])

        # unidet
        self.unidet_boxes = np.array([])
        self.unidet_classes = np.array([])
        self.unidet_scores = np.array([])

        self.unidet_active_boxes = np.array([])
        self.unidet_active_classes = np.array([])
        self.unidet_active_scores = np.array([])

        # shan
        self.shan_boxes = np.array([])
        self.shan_scores = np.array([])

    def __repr__(self) -> str:
        return f"{self.name}: {vars(self)}"

    def __str__(self) -> str:
        return f"{self.name}: {vars(self)}"

    def get_ground_truth(self, ground_truth_data, ontology):
        """
        Checks if image was skipped in the ground truth data, if not, then it gets the ground truth labels.
        """
        self.name = ground_truth_data["External ID"].split(".")[0]
        self.gt_boxes, self.gt_classes = [], []

        if not ground_truth_data["Skipped"]:
            for gt in ground_truth_data["Label"]["objects"]:
                try:
                    bbox = [
                        gt["bbox"]["left"],
                        gt["bbox"]["top"],
                        gt["bbox"]["left"] + gt["bbox"]["width"],
                        gt["bbox"]["top"] + gt["bbox"]["height"],
                    ]
                    self.gt_boxes.append(bbox)
                    self.gt_classes.append(ontology[gt["title"]])
                except ValueError as e:
                    raise ValueError(
                        f'Error in {self.name} -- {gt["title"]} not found in ontology.'
                    ) from e

        self.gt_boxes, self.gt_classes = np.array(self.gt_boxes, dtype="int"), np.array(
            self.gt_classes
        )

    def get_unidet_results(self, unidet_data):
        """
        Get the unidet predictions for the image.
        """
        self.unidet_boxes, self.unidet_classes, self.unidet_scores = (
            np.array(unidet_data["boxes"], dtype=int),
            np.array(unidet_data["classes"]),
            np.array(unidet_data["scores"]),
        )

    def get_shan_results(self, shan_data):
        """
        Get the shan predictions for the image.

        args: shan_data (dict) in format:
        {
            objects: [[boxes(4), score(1), state(1), offset_vector(3), left/right(1)], ...]],
            hands: [[boxes(4), score(1), state(1), offset_vector(3), left/right(1)], ...]],
        }
        """
        shan_boxes, shan_scores = self.parse_shan_results(shan_data)
        self.shan_boxes = np.array(shan_boxes, dtype=int)
        self.shan_scores = np.array(shan_scores)

    def parse_shan_results(self, shan_data):
        """
        Parse the shan predictions for the image.

        args: shan_data (dict) in format:
        {
            objects: [[boxes(4), score(1), state(1), offset_vector(3), left/right(1)], ...]],
            hands: [[boxes(4), score(1), state(1), offset_vector(3), left/right(1)], ...]],
        }

        returns: shan_boxes (list), shan_scores (list)
        """
        shan_boxes, shan_scores = [], []

        if shan_data["objects"] is not None:
            for obj in shan_data["objects"]:
                shan_boxes.append(obj[:4])
                shan_scores.append(obj[4])

        return shan_boxes, shan_scores

    def detect_active_objects(self, threshold=0.75):
        """
        Detects the active objects in the ground truth and unidet predictions by using the shan model as "ground truth".
        """
        gt_active = np.zeros(len(self.gt_boxes))
        unidet_active = np.zeros(len(self.unidet_boxes))

        if len(self.shan_boxes) > 0:

            shan_boxes = torch.from_numpy(self.shan_boxes)

            if len(self.gt_boxes) > 0:
                gt_boxes = torch.from_numpy(self.gt_boxes)
                ious_gt = torchvision.ops.box_iou(gt_boxes, shan_boxes).numpy()
                gt_threshold = np.argwhere(ious_gt > threshold)[:, 0]
                gt_active[gt_threshold] = 1

                self.gt_active_boxes = np.array(self.gt_boxes[gt_active == 1])
                self.gt_active_classes = np.array(self.gt_classes[gt_active == 1])

            else:
                self.gt_active_boxes = np.array([])
                self.gt_active_classes = np.array([])

            if len(self.unidet_boxes) > 0:
                unidet_boxes = torch.from_numpy(self.unidet_boxes)
                ious_unidet = torchvision.ops.box_iou(unidet_boxes, shan_boxes).numpy()
                unidet_threshold = np.argwhere(ious_unidet > threshold)[:, 0]
                unidet_active[unidet_threshold] = 1

                self.unidet_active_boxes = np.array(
                    self.unidet_boxes[unidet_active == 1]
                )
                self.unidet_active_classes = np.array(
                    self.unidet_classes[unidet_active == 1]
                )
                self.unidet_active_scores = np.array(
                    self.unidet_scores[unidet_active == 1]
                )
            else:
                self.unidet_active_boxes = np.array([])
                self.unidet_active_classes = np.array([])
                self.unidet_active_scores = np.array([])

    def remap_classes(self, mapper):
        """
        Remaps the classes to the new ontology (i.e. combined categories).

        args: mapper (dict) in format:
        {
            new_class: [old_class1, old_class2, ...],
            ...
        }
        """

        for new_class, old_classes in mapper.items():
            try:
                if len(self.gt_classes) > 0:
                    self.gt_classes[np.isin(self.gt_classes, old_classes)] = new_class

                if len(self.gt_active_classes) > 0:
                    self.gt_active_classes[
                        np.isin(self.gt_active_classes, old_classes)
                    ] = new_class

                if len(self.unidet_classes) > 0:
                    self.unidet_classes[
                        np.isin(self.unidet_classes, old_classes)
                    ] = new_class

                if len(self.unidet_active_classes) > 0:
                    self.unidet_active_classes[
                        np.isin(self.unidet_active_classes, old_classes)
                    ] = new_class

            except:
                warnings.warn(f"Warning: No classes remapped for {self.name}.")
                continue
