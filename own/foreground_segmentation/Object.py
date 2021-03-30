#!/usr/bin/env python3

import numpy as np

class Object():
    """
    Class to represent objects identified in a frame.
    """
    def __init__(self, box_width, box_height, pos_x, pos_y, silhouette, center_x=0, center_y=0, obj_id=0, vel_x=0, vel_y=0):
        """
        Parameters
        ----------
        box_width, box_height : int
            Dimensions of box that bounds object
        pos_x, pos_y : int
            Position of bounding box's top left pixel.
        object_id : [Object]
            Object's unique id.
        silhouette : [int, int, (int, int, int)]
            A 3d array as big as the object's bounding box containing the rgb image of the object's silhouette.
        """
        
        self.id = obj_id
        self.box_width = box_width
        self.box_height = box_height
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.center_x = center_x
        self.center_y = center_y
        self.state = np.array([[self.pos_x], [self.pos_y], [vel_x], [vel_y]])
        self.predicted_state = np.array([[self.pos_x], [self.pos_y], [vel_x], [vel_y]])
        self.silhouette = silhouette
        self.missing = True
        self.missing_count = 0
        self.matched_id = -1


    def contains(self, point):
        """Takes in a tuple (row, col) coordinate. Returns True if point is within self"""
        return ((point[0] >= self.pos_y and point[0] <= self.pos_y + self.box_height) \
            and (point[1] >= self.pos_x and point[1] <= self.pos_x + self.box_width))

