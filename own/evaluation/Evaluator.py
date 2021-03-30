#!/usr/bin/env python3
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import os
from scipy.optimize import linear_sum_assignment

from foreground_segmentation.Object import Object
from data_reading.FrameList import FrameList
from tracker.ObjectMatcher import ObjectMatcher

class Evaluator():
    """
    Class that evaluates performance of the pipeline
    """
    def __init__(self, frame_list, gt_csv_path):
        """
        Parameters
        ----------
        frame_list : [Frame, ...]
            1D array of Frames
        gt_csv : [<frame_count>, <obj_id>, <bb_x>, <bb_y>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>, ...]
            Ground truth csv file
        """
        self.frame_list = frame_list
        self.gt_csv_path = gt_csv_path
        self.gt_frame_list = []
        
    
    def read_gt(self, downsampling_factor=1):
        """Creates a FrameList with the ground truth"""
        video_path = self.frame_list.video_path
        self.gt_frame_list = FrameList(video_path)
        self.gt_frame_list.read_frames()

        with open(self.gt_csv_path) as gt_csv_file:
            csv_reader = csv.reader(gt_csv_file, delimiter=',')
            for row in csv_reader:
                # Retrieve useful information
                gt_fc = int(row[0]) - 1
                gt_obj_id = int(row[1])
                gt_bb_x = int(row[2])//downsampling_factor
                gt_bb_y = int(row[3])//downsampling_factor
                gt_bb_width = int(row[4])//downsampling_factor
                gt_bb_height = int(row[5])//downsampling_factor
                
                # Create object
                obj = Object(gt_bb_width, gt_bb_height, gt_bb_x, gt_bb_y, [], obj_id=gt_obj_id)
                self.gt_frame_list.frames[gt_fc].add_object(obj)

    def generate_eval_csv(self, sequence, downsampling_factor=1):
        seqs = ['02', '04', '09']
        if sequence not in seqs:
            print('ERROR_evaluation: Wrong sequence name')
            return

        if os.path.basename(os.getcwd()) != 'own':
            print('ERROR_evaluation: The current directory should be \"own\"')
            return

        with open(f'{os.path.dirname(os.getcwd())}/Evaluation/{sequence}.csv', 'w', newline='') as eval_file:
            csv_writer = csv.writer(eval_file, delimiter=',')
            for frame in self.frame_list.frames:
                frame_number = frame.number
                if frame.object_list:
                    for obj in frame.object_list:
                        csv_writer.writerow([frame_number, obj.id, obj.pos_x*downsampling_factor, obj.pos_y*downsampling_factor, obj.box_width*downsampling_factor, obj.box_height*downsampling_factor])

    def generate_metrics(self):
        """
        True positive (TP): A detection that has at least 20% overlap with the associated ground truth bounding box.
        False Positive (FP): A detection that has less than 20% overlap with the associated ground truth bounding box,
            or that has no associated ground truth bounding box.
        False Negative (FN): A ground truth bounding box that has no associated detection,
            or for which the associated detection overlap by less than 20%
        """
        tp_sum = 0
        fp_sum = 0
        fn_sum = 0
        avg_tp_overlap_list = []
        id_switches = 0
        prev_gt_ids = []
        
        #identity_switches = 0
        
        for frame_index in range(len(self.gt_frame_list.frames)):
            jaccard_table = self.create_jaccard_table(self.gt_frame_list.frames[frame_index], self.frame_list.frames[frame_index])
            gt_idxs, det_idxs = linear_sum_assignment(jaccard_table, maximize=True)
            #print(f'Frame number: {frame_index}, gt_idx: {gt_idxs}, det_idx: {det_idxs}')
            if frame_index != 0:
                prev_gt_ids = np.array([prev_gt.id for prev_gt in self.gt_frame_list.frames[frame_index-1].object_list])
                #print(f'Prev_gt_ids: {prev_gt_ids}')
            for obj_idx in range(len(gt_idxs)):
                gt_idx = gt_idxs[obj_idx]
                det_idx = det_idxs[obj_idx]
                jaccard_idx = jaccard_table[gt_idx, det_idx]

                if jaccard_idx < 0.2:
                    fn_sum += 1
                    fp_sum += 1
                else:
                    gt_obj = self.gt_frame_list.frames[frame_index].object_list[gt_idx]
                    det_obj = self.frame_list.frames[frame_index].object_list[det_idx]
                    gt_obj.matched_id = det_obj.id

                    if prev_gt_ids.size != 0:
                        if np.where(prev_gt_ids == gt_obj.id)[0].size != 0:
                            #print(f'Before prev_matched_id: {np.where(prev_gt_ids == gt_obj.id)[0][0]}')
                            prev_matched_id = self.gt_frame_list.frames[frame_index-1].object_list[np.where(prev_gt_ids == gt_obj.id)[0][0]].matched_id
                            if prev_matched_id != det_obj.id:
                                id_switches += 1
                            else:
                                avg_tp_overlap_list.append(jaccard_idx)
                        
                    tp_sum += 1
            (num_gt, num_det) = jaccard_table.shape
            fp_sum += max(num_det-num_gt, 0)
            fn_sum += max(num_gt-num_det, 0)
        avg_tp_overlap = sum(avg_tp_overlap_list)/len(avg_tp_overlap_list)

        return tp_sum, fp_sum, fn_sum, id_switches, tp_sum/(tp_sum+fp_sum), tp_sum/(tp_sum+fn_sum), avg_tp_overlap
        
    def create_jaccard_table(self, gt_frame, det_frame):
        """
        gt_frame: Ground truths frame
        det_frame: Detections frame
        """
        obj_matcher = ObjectMatcher("simple")
        jaccard_table = np.zeros((len(gt_frame.object_list), len(det_frame.object_list)))

        for i in range(len(gt_frame.object_list)):
            for j in range(len(det_frame.object_list)):
                obj_1 = gt_frame.object_list[i]
                obj_2 = det_frame.object_list[j]
                objs_area = obj_1.box_width*obj_1.box_height + obj_2.box_width*obj_2.box_height
                jaccard_table[i, j] = obj_matcher.overlap_score(obj_1, obj_2)/objs_area
        
        return jaccard_table



    def visualize_gt(self):
        # This is currently not used, might aswell use TrackerVisualizer
        frames = self.gt_frame_list.frames
        tot_frames = len(frames)
        plt.figure()
        i = 0
        display_image = True
        while display_image:
            plt.imshow(cv2.cvtColor(frames[i].rgb_image, cv2.COLOR_BGR2RGB))
            ax = plt.gca()
            for obj in frames[i].object_list:
                rect = patches.Rectangle((obj.pos_x, obj.pos_y), obj.box_width, obj.box_height, edgecolor='r', facecolor="none")
                ax.add_patch(rect)
            rect = patches.Rectangle((912, 484), 97, 109, edgecolor='r', facecolor="none")
            ax.add_patch(rect)
            plt.pause(0.0001)
            plt.clf()
            i += 5
            if i >= tot_frames:
                display_image = False

