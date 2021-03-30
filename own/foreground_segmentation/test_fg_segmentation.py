import os
import cv2
import sys
sys.path.append('..') 
from data_reading.Frame import Frame
from data_reading.FrameList import FrameList
from ForegroundSegmentation import ForegroundSegmentation
import matplotlib.pyplot as plt
import numpy as np


def load_image(img_set, img_n):
    # Loads an single image
    # img_set: string {chess, forward, scar}
    # img_n: integer
    if img_set == "low_noise" and (0 <= img_n <= 5):
        folder = "low_noise"
        img_type = ".png"
    elif img_set == "high_noise" and (0 <= img_n <= 5): 
        folder = "high_noise"
        img_type = ".png"
    else:
        print("ERROR in load_image")
        return -1
    img_path = os.path.join(os.getcwd(),'..', 'test_data', folder, f'img{img_n}{img_type}')
    print(img_path)
    return cv2.imread(img_path, 0)

def load_image_set(img_set):
    # Loads a set of images
    # img_set: string {low_noise,}
    if img_set == "low_noise":
        image_start = 0
        shape0, shape1 = load_image("low_noise", image_start).shape
        n_images = 6
    elif img_set == "high_noise":
        image_start = 0
        shape0, shape1 = load_image("high_noise", image_start).shape
        n_images = 6
    else:
        print("ERROR in load_set")
        return -1

    images = np.zeros((n_images, shape0, shape1))
    for i in range(n_images):
        images[i] = load_image(img_set, image_start+i)

    return images

def create_fake_frame_list(image_set):
    frame_list = FrameList('test')
    frame_count = 0
    for frame in img_set:
        frame_obj = Frame(frame_count, np.uint8(frame))
        frame_obj.likelihood_image = np.uint8(frame)
        frame_list.frames.append(frame_obj)
        frame_count += 1
    return frame_list

img_set = load_image_set('low_noise')
frame_list = create_fake_frame_list(img_set)
fg_seg = ForegroundSegmentation(frame_list)
fg_seg.remove_noise(erode_iter=1, dilate_iter=3)
fg_seg.label_image(min_box_area=100)
for frame in frame_list.frames:
    plt.figure()
    plt.imshow(frame.labeled_image)
    ax = plt.gca()
    for obj in frame.object_list:
        ax.add_patch(plt.Rectangle((obj.pos_x, obj.pos_y), obj.box_width, obj.box_height, fill=False, edgecolor='r'))

plt.show()