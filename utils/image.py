"""Object that stores the file path of the image, the ground truth labels, and shan/unidet predictions."""

import numpy as np
import pandas as pd # type: ignore
import torch # type: ignore
import torchvision # type: ignore


class ImageContainer:
    """
    Object that stores the file path of the image, the ground truth labels, and shan/unidet predictions.
    """

    def __init__(self) -> None:
        self.name: str = ''

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

    def get_ground_truth(self, ground_truth_data):
        """
        Checks if image was skipped in the ground truth data, if not, then it gets the ground truth labels.
        """
        self.name = ground_truth_data["External ID"].split(".")[0]
        self.gt_boxes, self.gt_classes = [], []

        if not ground_truth_data["Skipped"]:
            for gt in ground_truth_data["Label"]["objects"]:
                bbox = [
                    gt["bbox"]["left"],
                    gt["bbox"]["top"],
                    gt["bbox"]["left"] + gt["bbox"]["width"],
                    gt["bbox"]["top"] + gt["bbox"]["height"],
                ]
                self.gt_boxes.append(bbox)
                self.gt_classes.append(gt["title"])

        self.gt_boxes = np.array(self.gt_boxes, dtype="int")
        self.gt_classes = np.array(self.gt_classes)

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
    
    def remap_classes(self, mapper:dict):
        '''
        Remaps the classes of the ground truth and unidet predictions to the new classes.

        args: mapper (dict) in format:
        {
            label_id: {label: label, subclasses: [subclasses_ids], subclasses_names: [subclasses_names]},
            ...
        }

        Finds the current class in mapper's subclasses_names and replaces it with the label_id.
        Also saves label to self.gt_class_names and self.unidet_class_names.
        '''

        gt_classes = self.gt_classes.copy()
        unidet_classes = self.unidet_classes.copy()
        gt_active_classes = self.gt_active_classes.copy()
        unidet_active_classes = self.unidet_active_classes.copy()
        
        for i, gt_class in enumerate(gt_classes):
            for key, value in mapper.items():
                if gt_class in value['subclasses_names']:
                    gt_classes[i] = key

        for i, unidet_class in enumerate(unidet_classes):
            for key, value in mapper.items():
                if unidet_class in value['subclasses']:
                    unidet_classes[i] = key

        for i, gt_active_class in enumerate(gt_active_classes):
            for key, value in mapper.items():
                if gt_active_class in value['subclasses_names']:
                    gt_active_classes[i] = key

        for i, unidet_active_class in enumerate(unidet_active_classes):
            for key, value in mapper.items():
                if unidet_active_class in value['subclasses']:
                    unidet_active_classes[i] = key
        
        self.gt_classes = np.array(gt_classes, dtype=np.int32)
        self.unidet_classes = np.array(unidet_classes, dtype=np.int32)
        self.gt_active_classes = np.array(gt_active_classes, dtype=np.int32)
        self.unidet_active_classes = np.array(unidet_active_classes, dtype=np.int32)

    def del_human_preds(self):
        '''
        Delete all unidet boxes, classes and scores corresponding to class = 0 (human).
        '''

        # get indexes where unidet_classes == 0
        indexes = np.argwhere(self.unidet_classes == 0)

        # delete all boxes, classes and scores corresponding to indexes
        self.unidet_boxes = np.delete(self.unidet_boxes, indexes, axis=0)
        self.unidet_classes = np.delete(self.unidet_classes, indexes, axis=0)
        self.unidet_scores = np.delete(self.unidet_scores, indexes, axis=0)

        # get indexes where unidet_active_classes == 0
        indexes = np.argwhere(self.unidet_active_classes == 0)

        # delete all boxes, classes and scores corresponding to indexes
        self.unidet_active_boxes = np.delete(self.unidet_active_boxes, indexes, axis=0)
        self.unidet_active_classes = np.delete(self.unidet_active_classes, indexes, axis=0)
        self.unidet_active_scores = np.delete(self.unidet_active_scores, indexes, axis=0)


def create_mapping_dict(xls_path:str, sheet_name:str='LabelSpace'):
    '''
    Creates a mapping dictionary from the xls file.
    
    args: xls_path (str) path to the xls file

    xls format:
    column 1: label (str)
    column 2: label_id (int)
    column 3: subclasses (comma separated)
    column 4: subclasses_ids (comma separated)

    returns: mapper (dict) in format:
    {
        label_id: {label: label, subclasses: [subclasses_ids], subclasses_names: [subclasses_names]},
        ...
    }
    '''

    mapper = {}

    df = pd.read_excel(xls_path, sheet_name=sheet_name)
    df = df.dropna(subset=['ID'])

    for i, row in df.iterrows():
        label = row['Label']
        label_id = int(row['ID'])
        subclasses = row['Subclasses']
        subclasses_ids = row['Subclass ID']

        if isinstance(subclasses, str):
            # print all fields
            print(f'Label: {label},\n label_id: {label_id},\n subclasses: {subclasses},\n subclass_ids:{subclasses_ids} \n \n')

            # if subclasses_ids is a float then convert to str without decimal
            if isinstance(subclasses_ids, float):
                subclasses_ids = str(int(subclasses_ids))

            subclasses = subclasses.split(',')
            subclasses_ids = subclasses_ids.split(',')
            subclasses_ids = [int(x) for x in subclasses_ids]

        mapper[label_id] = {'label': label, 'subclasses': subclasses_ids, 'subclasses_names': subclasses}

    return mapper