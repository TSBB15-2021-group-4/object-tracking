#!/usr/bin/env python3

import cv2
import numpy as np
from .Frame import Frame

class FrameList():
    """
    Class to represent a list of frames (image sequence).
    """
    def __init__(self, video_path):
        """
        Parameters
        ----------
        frames : [Frame, Frame, ....]
            1D array of frames
        video_path : str
            String containing the relative path to the video sequence
        """

        self.frames = []
        self.video_path = video_path

    def read_frames(self):
        """Read data from video sequence, build self.frames"""
        
        cap = cv2.VideoCapture(self.video_path) 

        if (cap.isOpened() == False):
            print("ERROR: Unable to open video. The file path is likely wrong!")

        fc = 0
        ret = True

        while (ret):
            # Capture the video frame by frame
            ret, frame = cap.read()
            if ret:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
                self.frames.append(Frame(number=fc, rgb_image=frame, gray_image=gray_frame))
                fc += 1
                # Uncomment the following 3 lines if you don't want to see the image sequence
                #if cv2.waitKey(1) & 0xFF == ord('q'): 
                #    break
                #cv2.imshow('Frame', gray_frame)

        cap.release()
