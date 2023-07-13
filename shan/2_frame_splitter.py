# split video into frames

import cv2
import os
import tqdm


class Video:
    def __init__(self, root, file):
        self.root = root
        self.file = file
        self.video = os.path.join(root, file)

    def frame_split(self, fps):
        cap = cv2.VideoCapture(self.video)
        fps_original = int(cap.get(cv2.CAP_PROP_FPS))
        downsample = fps_original // fps
        idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % downsample == 0:
                outpath = os.path.join(
                    self.root, self.file.split(".")[0], f"frame_{idx}.jpg"
                )
                cv2.imwrite(outpath, frame)
            idx += 1

        cap.release()
        cv2.destroyAllWindows()

# change this line to the path of the folder containing the videos
filepath = "/workspaces/cdss-modeling/shan/videos/P-02/subclips"

os.chdir(filepath)

# for each video in filepath, create a folder with the same name as the video without the file extension and save the frames in that folder

for file in tqdm.tqdm(os.listdir(filepath)):
    if file.endswith(".MP4"):
        # make folder for frames
        if not os.path.exists(file.split(".")[0]):
            os.mkdir(file.split(".")[0])

        # split video into frames and save
        vid = Video(filepath, file)
        vid.frame_split(2)
