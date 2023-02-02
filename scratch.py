"""This is a scratch file for testing out code snippets."""

import pickle
import json

from utils.dataset import HomeDataEvalSubset

# set root directory
root = "datasets/home-data-eval/"

# load mapper from pickle file
mapper = pickle.load(open("datasets/mapper.pkl", "rb"))

# load annotations from json file
annotations = json.load(open("datasets/ground_truth_labels.json"))

# create dataset
data = HomeDataEvalSubset(root, mapper, annotations)

# test dataset
print(data[0])
