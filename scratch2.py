# sum the length of video clips in a list

import os
from moviepy.editor import VideoFileClip
from math import floor

def get_video_length(video_path):
    """
    get length of video in minutes
    """
    clip = VideoFileClip(video_path)
    return int(floor(clip.duration))/60

def get_video_lengths(video_paths):
    """
    get length of all videos in a folder and sum them
    """
    return sum([get_video_length(video_path) for video_path in video_paths])

nov13 = [
    'shan/videos/P-03/GOPR0501.MP4',
    'shan/videos/P-03/GOPR0502.MP4',
    'shan/videos/P-03/GOPR0503.MP4',
    'shan/videos/P-03/GP010503.MP4',
    'shan/videos/P-03/GOPR0504.MP4',
]

nov14 = [
    'shan/videos/P-03/GOPR0505.MP4',
    'shan/videos/P-03/GOPR0506.MP4',
    'shan/videos/P-03/GOPR0507.MP4',
    'shan/videos/P-03/GOPR0508.MP4',
    'shan/videos/P-03/GOPR0509.MP4',
    'shan/videos/P-03/GOPR0510.MP4',
]

nov17 = [
    'shan/videos/P-03/GOPR0511.MP4',
    'shan/videos/P-03/GOPR0512.MP4',
    'shan/videos/P-03/GOPR0513.MP4',
    'shan/videos/P-03/GOPR0514.MP4',
    'shan/videos/P-03/GOPR0515.MP4',
    'shan/videos/P-03/GOPR0516.MP4',
]

nov23 = [
    'shan/videos/P-03/GOPR0517.MP4',
    'shan/videos/P-03/GOPR0518.MP4',
    'shan/videos/P-03/GOPR0519.MP4',
    'shan/videos/P-03/GOPR0520.MP4',
    'shan/videos/P-03/GOPR0521.MP4',
]

dec17 = [
    'shan/videos/P-03/GOPR0522.MP4',
    'shan/videos/P-03/GOPR0776.MP4',
    'shan/videos/P-03/GOPR0777.MP4',
    'shan/videos/P-03/GP010777.MP4',
]

dec18 = [
    'shan/videos/P-03/GOPR0778.MP4',
    'shan/videos/P-03/GOPR0779.MP4',
    'shan/videos/P-03/GOPR0780.MP4',
    'shan/videos/P-03/GP010780.MP4',
    'shan/videos/P-03/GOPR0781.MP4',
    'shan/videos/P-03/GOPR0782.MP4',
]

# run get_video_lengths on each list and print the sum

print(get_video_lengths(nov13))
print(get_video_lengths(nov14))
print(get_video_lengths(nov17))
print(get_video_lengths(nov23))
print(get_video_lengths(dec17))
print(get_video_lengths(dec18))
