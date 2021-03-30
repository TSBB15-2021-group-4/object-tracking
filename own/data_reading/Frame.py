#!/usr/bin/env python3

class Frame():
    """
    Class to represent a frame (image).
    """
    def __init__(self, number, rgb_image, gray_image):
        """
        Parameters
        ----------
        number : int
            Index of this frame in the list of frames
        rgb_image : [int, int, (int, int, int)]
            5D array containing color of each pixel
        gray_image : [int, int, int]
            3D array containing image in gray scale
        object_list : [Object]
            1D array of objects identified in this specific frame
        binary_image : [int, int]
            3D array where 0s represent background and 1s foreground
        labeled_image : [int, int]
            2D array where the pixel values represent an object
        """

        self.number = number
        self.rgb_image = rgb_image 
        self.gray_image = gray_image
        self.object_list = []
        self.binary_image = None
        self.labeled_image = []


    def add_object(self, obj):
        self.object_list.append(obj)


    def remove_object(self, obj):
        if obj in self.object_list:
            self.object_list.remove(obj)
        else:
            print('WARNING: No object was removed')