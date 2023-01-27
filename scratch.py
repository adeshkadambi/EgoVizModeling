"""This is a scratch file for testing out code snippets."""

import numpy as np
import pandas as pd # type: ignore
import pickle
import draw
import cv2 # type: ignore

# test draw module using images in datasets/home-data-eval folder

# # load image containers from pickle file
# with open("datasets/home-data-eval/results_remapped.pkl", "rb") as f:
#     images = pickle.load(f)

# print('Loaded Data')

# # load mapper from pickle file
# with open("datasets/home-data-eval/mapper.pkl", "rb") as f:
#     mapper = pickle.load(f)

# # # remap classes
# # for img in images:
# #   img.remap_classes(mapper['Subclass ID'])

# # # save remapped classes to pickle file
# # with open("datasets/home-data-eval/results_remapped.pkl", "wb") as f:
# #     pickle.dump(images, f)

# print('Loaded Mapper & Remapped Classes')

# # draw bounding boxes on images and save to tests/ folder
# imgs = draw.batch_draw_bbox_image_container(images[:1], "datasets/home-data-eval", mapper, "/workspaces/cdss-modeling/tests")

# # plot images

# cv2.imshow('image', imgs[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# random numpy array of 1 and 0
a = np.random.randint(2, size=(4))
# random numpy array of letters
b = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), size=(4))

print(a)
print(b)

print(a[a != 0])
print(b[a != 0])