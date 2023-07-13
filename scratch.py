"""This is a scratch file for testing out code snippets."""

import pickle
import json

from utils.dataset import HomeDataEvalSubset
from torch.utils.data import DataLoader  # type: ignore

# set root directory
root = "datasets/home-data-eval/"

# load mapper from pickle file
mapper = pickle.load(open("datasets/mapper.pkl", "rb"))

# load annotations from json file
annotations = json.load(open("datasets/ground_truth_labels.json"))

# create dataset
data = HomeDataEvalSubset(root, mapper, annotations)

# test dataset
dataloader = DataLoader(data, batch_size=1, shuffle=True)

img, target = next(iter(dataloader))

print(img.shape)
print(target)
