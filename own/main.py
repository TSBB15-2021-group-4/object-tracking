#!/usr/bin/env python3

from data_reading.FrameList import FrameList
from background_model.BackgroundModel import BackgroundModel
from foreground_segmentation.ForegroundSegmentation import ForegroundSegmentation
from tracker.Tracker import Tracker
from tracker.ObjectMatcher import ObjectMatcher
from matplotlib import pyplot as plt
from tracker.Predictor import Predictor
from tracker.TrackerVisualizer import TrackerVisualizer
from evaluation.Evaluator import Evaluator
import numpy as np
import time


def main():
    """
    Track objects.
    """
    visualize_steps = True
    evaluate = True
    downsampling_factor = 2
    mot_sequence = '02' #02, 04 or 09
    # ---------------- DATA READING -------------------
    print('Data reading...')
    # Read image sequence into frames list
    video_path = f'test_data/MOT17-{mot_sequence}-raw.webm'
    frame_list = FrameList(video_path)
    frame_list.read_frames()
    
    # ------------- BACKGROUND MODELING ---------------
    print('Background modeling...')
    # For each frame in frame_list.frames, create likelihood images & suppress shadows.
    filter_type = 'gaussian'
    background_model = BackgroundModel(filter_type)
    background_model.create_binary_images(frame_list.frames, visualize=visualize_steps)
    background_model.suppress_shadows(frame_list.frames, visualize=visualize_steps)
    plt.close('all')

    # ------------- FOREGROUND SEGMENTATION -----------
    print('Foreground segmentation...')
    # For each frames in frame_list.frames, remove noise & create objects
    foreground_segmentation = ForegroundSegmentation(frame_list)
    foreground_segmentation.remove_noise()
    foreground_segmentation.label_image(min_box_area=1500, visualize=visualize_steps)

    # ------------------- TRACKER ---------------------
    print('Tracking...')
    # Create ObjectMatcher and Predictor objects used by Tracker
    object_matcher = ObjectMatcher(match_algorithm='simple')
    predictor = Predictor(Q0=1, R0=1, P0=np.array([1, 1, 0.01, 0.01]), T=1)

    # Create Tracker object
    tracker = Tracker(object_matcher, predictor)

    plt.figure()
    # Match objects between frames 2 by 2
    print('Number of frames =', len(frame_list.frames))
    for i in range(1, len(frame_list.frames) - 1): 
        tracker.match_objects(frame_list.frames[i], frame_list.frames[i + 1])
        if (i % 20 == 0):
            print(round(i/len(frame_list.frames) * 100), '%')   
    
    plt.close('all')
    TrackerVisualizer(frame_list)

    # ------------------ EVALUATION -------------------
    print('Evaluating...')
    # This will only work with MOT17 dataset 02, 04 or 09
    if(video_path.startswith('test_data/MOT17') and evaluate):
        evaluator = Evaluator(frame_list, f'test_data/MOT17-{mot_sequence}-gt.txt')
        evaluator.read_gt(downsampling_factor=downsampling_factor)
        #evaluator.generate_eval_csv(mot_sequence, downsampling_factor=downsampling_factor)
        tp_sum, fp_sum, fn_sum, id_switches, precision, recall, avg_tp_overlap = evaluator.generate_metrics()
        print(f'tp_sum: {tp_sum}')
        print(f'fn_sum: {fn_sum}')
        print(f'fp_sum: {fp_sum}')
        print(f'id_switches: {id_switches}')
        print(f'precision: {precision}')
        print(f'recall: {recall}')
        print(f'avg_tp_overlap: {avg_tp_overlap}')


        # Visualize ground truth
        plt.close('all')
        #TrackerVisualizer(evaluator.gt_frame_list)


if __name__ == '__main__':
    main()