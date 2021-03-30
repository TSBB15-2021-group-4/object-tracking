#!/usr/bin/env python3
import cv2
import numpy as np
from foreground_segmentation.Object import Object
import matplotlib.pyplot as plt
from matplotlib import patches

class ForegroundSegmentation():
    """
    Class that removes noise and labels objects in binary images
    """
    def __init__(self, frame_list):
        """
        Parameters
        ----------
        frame_list : [Frame, ...]
            1D array of Frames
        """
        self.frame_list = frame_list
        self.obj_counter = 0
        self.visualize_silhouettes = False
        
    
    def remove_noise(self, struc_elem_size=(3,3), erode_iter=1, dilate_iter=1):
        """Uses morphological operations to remove noise"""
        struc_elem = np.ones(struc_elem_size)
        for frame in self.frame_list.frames:
            frame.binary_image = cv2.dilate(frame.binary_image, struc_elem, iterations=2)
            frame.binary_image = cv2.erode(frame.binary_image, struc_elem, iterations=2)
            frame.binary_image = cv2.erode(frame.binary_image, struc_elem, iterations=2)
            frame.binary_image = cv2.dilate(frame.binary_image, struc_elem, iterations=2)
            frame.binary_image = cv2.dilate(frame.binary_image, struc_elem, iterations=1)
            frame.binary_image = cv2.erode(frame.binary_image, struc_elem, iterations=1)
            frame.binary_image = cv2.erode(frame.binary_image, struc_elem, iterations=1)
            frame.binary_image = cv2.dilate(frame.binary_image, struc_elem, iterations=1)

    def label_image(self, min_box_area=100, visualize=False):
        """Finds and sets the obj.id for each found object obj"""
        
        if self.visualize_silhouettes:
            _, (ax1, ax2, ax3) = plt.subplots(1, 3)

        for frame in self.frame_list.frames:
            '''
            The structure of bound_box_info is:
                Leftmost x coordinate,
                Topmost y coordinate,
                Width,
                Height,
                Area
            '''
            num_of_objs, labeled_img, bound_box_info, obj_center_points = cv2.connectedComponentsWithStats(frame.binary_image)
            frame.labeled_image = labeled_img

            # Find index of the bounding box with the largest area(background)
            background_index = np.argmax(bound_box_info, axis=0)[4]

            for obj_index in range(num_of_objs):
                box_area = bound_box_info[obj_index, 4]
                # Filter out the background and objects that have an area lower than min_box_area
                if obj_index == background_index or box_area < min_box_area:
                    continue

                box_x = bound_box_info[obj_index, 0]
                box_y = bound_box_info[obj_index, 1]
                box_width = bound_box_info[obj_index, 2]
                box_height = bound_box_info[obj_index, 3]

                box_center_x = obj_center_points[obj_index, 0]
                box_center_y = obj_center_points[obj_index, 1]

                silhouette = frame.binary_image[box_y:box_y+box_height, box_x:box_x+box_width, None] * \
                             frame.rgb_image[box_y:box_y+box_height, box_x:box_x+box_width, :]

                # To visualize silhouettes:
                if self.visualize_silhouettes:
                    ax1.imshow(frame.binary_image[box_y:box_y+box_height, box_x:box_x+box_width, None], 'gray')
                    ax2.imshow(frame.rgb_image[box_y:box_y+box_height, box_x:box_x+box_width, :])
                    ax3.imshow(silhouette)
                    plt.pause(0.5)
                
                obj = Object(box_width, box_height, box_x, box_y, silhouette, center_x=box_center_x, center_y=box_center_y, obj_id=self.obj_counter)
                self.obj_counter += 1

                frame.add_object(obj)
        if visualize:
            self.visualize_labeling(self.frame_list.frames)
        

    def visualize_labeling(self, frames):
        tot_frames = len(frames)
        plt.figure()
        i = 0
        display_image = True
        while display_image:
            plt.imshow(frames[i].labeled_image)
            ax = plt.gca()
            for obj in frames[i].object_list:
                rect = patches.Rectangle((obj.pos_x, obj.pos_y), obj.box_width, obj.box_height, edgecolor='r', facecolor="none")
                ax.add_patch(rect)
            plt.pause(0.0001)
            plt.clf()
            i += 5
            if i >= tot_frames:
                display_image = False