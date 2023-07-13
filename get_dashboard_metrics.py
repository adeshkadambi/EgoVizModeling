"""This module contains code for generating metrics for the dashboard."""

import os
import pickle
import re
import tqdm
import json

from typing import Dict, List
from dashboard_video_key import video_key_p03 as video_key
from dashboard_metrics import (
    interaction_percentage,
    interactions_per_hour,
    average_interaction_duration,
)

# sort alphanumeric strings
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


# for each entry in video_key, create a dictionary to hold metrics and frames for each day
metrics: Dict[str, Dict[str, float]] = {}
frames: Dict[str, List] = {}

for date, videos in video_key.items():
    metrics[date] = {}
    frames[date] = []

# root directory from user input
root = input("Enter the path to root dir (i.e., subclips_shan): ")

# pre-sort the list of subdirectories
subdirs = sorted_alphanumeric(os.listdir(root))

# for each subdirectory in root, get the date by checking if the name is in the video_key dict
for subdir in tqdm.tqdm(subdirs):
    # if any of the lists in video_key are a substring of subdir, get the date
    for date, videos in video_key.items():
        if any(video in subdir for video in videos):
            # get the date
            video_date = date

    # load all .pkl files in the subdirectory
    for file in sorted_alphanumeric(os.listdir(os.path.join(root, subdir))):
        if file.endswith(".pkl"):
            # load the file and append to frames
            try:
              with open(os.path.join(root, subdir, file), "rb") as p:
                  frames[video_date].append(pickle.load(p))
            except EOFError:
              print(f'{os.path.join(root, subdir, file)} is corrupt') # type: ignore

# for each date in frames, compute metrics and store in metrics
for date, frames_list in frames.items():
    metrics[date]["interaction_percentage"] = interaction_percentage(frames_list)
    metrics[date]["interactions_per_hour"] = interactions_per_hour(frames_list)
    metrics[date]["average_interaction_duration"] = average_interaction_duration(
        frames_list
    )

# save metrics to json file and log to console
print(metrics)

with open("/workspaces/cdss-modeling/aim1_data/P-03/dashboard_metrics.json", "w") as f: # type: ignore
    json.dump(metrics, f)
