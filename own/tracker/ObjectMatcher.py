#!/usr/bin/env python3

import numpy as np
import cv2
import statistics

class ObjectMatcher():
    """
    Class that is used for doing object macthing based on a score of overlaps.
    """
    def __init__(self, match_algorithm):
        """
        Parameters
        ----------
        type : str
            'simple' f0r simple overlap score matching, 'hungarian ' for Hungarian methods.
        """
        self.match_algorithm = match_algorithm

        # Initiate SIFT detector
        self.orb = cv2.ORB_create()
        # Create BFMatcher object
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


    def match_objects(self, frame_i, frame_j, max_missing_frame_count = 10):
        """
        Compare objects in frame_i to objects in frame_j and assign corresponding
        ids to the objects that are the same.
        Parameters to tune:
            max_missing_frame_count : int
                how many frames a bounding box can be missing before forgotten
            score_table_threshold : float
                how sensetive the object popper should be to objects merging
                (Ex 0.3 means object pops if two or more objects have 30% match rate)

            The row
            np.sum(bin_crop) / (obj.box_width*obj.box_height) > 0.2
            here it is 20% filled box or we forget it.
        """

        scores_table = self.create_scores_table(frame_i, frame_j)

        # Pops new big objects that seemingly merge two old objects
        score_table_threshold = 0.3
        # normalize scores_table columnwise
        for col in range(scores_table.shape[1]):
            col_sum = sum(scores_table[:,col])
            if col_sum > 0:
                scores_table[:,col] /= col_sum
        scores_thresh = scores_table > score_table_threshold
        scores_thresh_sum = scores_thresh.sum(axis=0)
        popped_objects = 0
        for i in range(len(scores_thresh_sum)):
            if scores_thresh_sum[i] >= 2:
                frame_j.object_list.pop(i-popped_objects)
                popped_objects += 1

                # find the box-merged objects to prevent missing_count
                obj_ind = np.transpose(scores_thresh[:,i].nonzero())
                for ind in obj_ind:
                    ind = ind[0]
                    # will count as missing but do not want to up the missing_count
                    frame_i.object_list[ind].missing_count -= 1

        # update with popped big objects
        scores_table = self.create_scores_table(frame_i, frame_j)

        if len(scores_table) > 0:
            max_score = np.argwhere(scores_table == np.amax(scores_table, axis=0))
            # Assign objects ids

            for index in max_score:
                # Copy over object id's only
                if scores_table[index[0],index[1]] > 0:
                    frame_j.object_list[index[1]].id = frame_i.object_list[index[0]].id
                    frame_j.object_list[index[1]].state = frame_i.object_list[index[0]].state
                    frame_j.object_list[index[1]].predicted_state = frame_i.object_list[index[0]].predicted_state

                    frame_i.object_list[index[0]].missing = False

                    # if id given to new obj, 0 the whole row
                    scores_table[index[0],:] *= 0

        # add missing objects
        # get keypoints and matches
        keypoints_i, descriptors_i = self.orb.detectAndCompute(frame_i.rgb_image, None)
        keypoints_j, descriptors_j = self.orb.detectAndCompute(frame_j.rgb_image, None)
        matches = self.bf.match(descriptors_i, descriptors_j)

        # Filter matches so that all matches are from one existing foregound object to another
        # for missing object
        for obj in frame_i.object_list:
            if obj.missing:
                obj.missing_count += 1

                in_bb_matches = self.filter_matches(matches, frame_i, frame_j, keypoints_i, keypoints_j, obj)

                if len(in_bb_matches) > 0:
                    movement_x_list = []
                    movement_y_list = []
                    for match in in_bb_matches:
                        point_i = np.flip(keypoints_i[match.queryIdx].pt)
                        point_j = np.flip(keypoints_j[match.trainIdx].pt)
                        movement_vec = point_j-point_i
                        movement_y_list.append(movement_vec[0])
                        movement_x_list.append(movement_vec[1])

                    # take median to prevent outliars
                    median_movement_x = statistics.median(movement_x_list)
                    median_movement_y = statistics.median(movement_y_list)

                    # using the obj state since it is not a predicted value after using predictor
                    obj.state[0] += median_movement_x
                    obj.state[1] += median_movement_y

                    obj.pos_x = obj.state[0,0]
                    obj.pos_y = obj.state[1,0]

                    if obj.missing_count < max_missing_frame_count:
                        frame_j.add_object(obj)

    def create_scores_table(self, frame_i, frame_j):
        """Create scores table"""
        scores_table = np.zeros((len(frame_i.object_list), len(frame_j.object_list)))

        for i in range(len(frame_i.object_list)):
            for j in range(len(frame_j.object_list)):
                obj_1 = frame_i.object_list[i]
                obj_2 = frame_j.object_list[j]
                scores_table[i, j] = self.overlap_score(obj_1, obj_2)

        return scores_table


    def overlap_score(self, obj_1, obj_2):
        """Count how many pixels from two objects' bounding box overlap."""

        init_row = max(obj_1.pos_y, obj_2.pos_y)
        end_row = min(obj_1.pos_y + obj_1.box_height, obj_2.pos_y + obj_2.box_height)

        init_column = max(obj_1.pos_x, obj_2.pos_x)
        end_column = min(obj_1.pos_x + obj_1.box_width, obj_2.pos_x + obj_2.box_width)

        if (init_column < end_column and init_row < end_row):
            # Return number of pixels in overlap region
            return (end_column - init_column) * (end_row - init_row)

        return 0

    def filter_matches(self, matches, frame_i, frame_j, keypoints_i, keypoints_j, obj):
        """Filter the existing point matches and only keep those that match objects to other objects
        inside the obj bounding box"""
        filtered_matches = []

        for match in matches:
            point_i = np.flip(keypoints_i[match.queryIdx].pt)
            point_j = np.flip(keypoints_j[match.trainIdx].pt)

            # If both points in match are foregrounds and inside box:
            if ((frame_i.binary_image[round(point_i[0]), round(point_i[1])] == 1) and \
                    (frame_j.binary_image[round(point_j[0]), round(point_j[1])] == 1)):
                if obj.state[0] <= point_i[1] <= obj.state[0] + obj.box_width and \
                        obj.state[1] <= point_i[0] <= obj.state[1] + obj.box_height:
                    filtered_matches.append(match)

        return filtered_matches
