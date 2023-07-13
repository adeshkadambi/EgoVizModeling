"""Make a PyTorch Dataset for the Home Data Eval Subset."""

import os
import torch  # type: ignore

from torchvision.io import read_image  # type: ignore
from torch.utils.data import Dataset  # type: ignore


class HomeDataEvalSubset(Dataset):
    def __init__(self, root: str, mapper: dict, annotations: list[dict]):
        self.root = root
        self.mapper = mapper
        self.annotations = annotations

        # get list of all images in annotations
        img_names = [img["External ID"] for img in annotations]

        # get path for each img in img_names in all subdirectories of root
        images: list[str] = []
        for root, _, files in os.walk(self.root):
            for file in files:
                if file in img_names:
                    images.append(os.path.join(root, file))

        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        # load image
        img_path = self.images[idx]
        img = read_image(img_path)

        # get annotation for img
        img_name = os.path.basename(img_path)
        annotation = [
            img for img in self.annotations if img["External ID"] == img_name
        ][0]

        # get boxes and labels
        boxes = []
        labels = []

        if not annotation["Skipped"]:
            for gt in annotation["Label"]["objects"]:
                box = [
                    gt["bbox"]["left"],
                    gt["bbox"]["top"],
                    gt["bbox"]["left"] + gt["bbox"]["width"],
                    gt["bbox"]["top"] + gt["bbox"]["height"],
                ]
                boxes.append(box)
                labels.append(gt["title"])

        # remap labels
        labels = self.remap_labels(labels, self.mapper)

        # generate target dict
        target = {}
        target["image_id"] = torch.tensor([idx])
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        # calculate area of each box, if there are no boxes then make empty tensor
        if len(boxes) > 0:
            area = []
            for box in boxes:
                area.append((box[2] - box[0]) * (box[3] - box[1]))
            target["area"] = torch.as_tensor(area, dtype=torch.float32)
        else:
            target["area"] = torch.as_tensor([], dtype=torch.float32)

        return img, target

    @staticmethod
    def remap_labels(labels: list[str], mapper: dict) -> list[int]:
        """Remap labels to match the labels in the mapper."""

        labels = labels.copy()

        for i, label in enumerate(labels):
            for key, value in mapper.items():
                if label in value["subclasses_names"]:
                    labels[i] = key

        return [int(label) for label in labels]
