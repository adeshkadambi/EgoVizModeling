"""
Created on Thu Nov 17 15:36:09 2022

@author: KadambiA
"""

import os

from moviepy.editor import VideoFileClip  # type: ignore
from math import floor


def segments(duration, n=60):
    """
    calculate start and end times for each subclip
    """
    lst = []
    for i in range(0, duration, n):
        lst.append((i, i + n))
    return lst


def save_subclips(video_path, root):
    """
    save subclips of a video
    """
    print("Current file: ", video_path)
    try:
        # load video
        clip = VideoFileClip(video_path)

        # get duration and start and end times
        times = segments(int(floor(clip.duration)))

        # make subclips and save
        for idx, (start, end) in enumerate(times):
            newclip = clip.subclip(start, end)

            # save videos to new folder named "subclips"
            clipname = os.path.join(
                root,
                "subclips",
                video_path.split(".")[0] + "--" + str(idx + 1) + ".MP4",
            )

            print("clipname: ", clipname)

            # save videos
            newclip.write_videofile(clipname, fps=10, audio=False)

        clip.close()
    except Exception as e:
        print("Error: ", e, " for file: ", video_path)


filepath = "/workspaces/cdss-modeling/aim1_data/P-04"

# make a new folder to save subclips within filepath if it doesn't exist
if not os.path.exists(os.path.join(filepath, "subclips")):
    os.mkdir(os.path.join(filepath, "subclips"))

# iterate through all videos in filepath
for file in os.listdir(filepath):
    if file.endswith(".MP4"):
        os.chdir(filepath)
        save_subclips(file, filepath)
