#!/usr/bin/env python3

from tracker.Predictor import Predictor
import numpy as np
import cv2
from matplotlib import pyplot as plt

# TODO: flytta två objekten och funktionerna

class Tracker():
    """
    Class ...
    """
    def __init__(self, object_matcher, predictor):
        """
        Parameters
        ----------
        object_matcher : ObjectMatcher
            Object with functions that do matching of objects
        predictor : Predictor
            Predictor used for doing predictions of objects' states
        """
        # Parameters for Kalman filter (noise variances and uncertainty in init pos)
        self.predictor = predictor
        self.object_matcher = object_matcher


    def match_objects(self, frame_i, frame_j):
        """Identify objects from frame_i in frame_j and assign corresponding ids"""
        
        # Predict position of objects in frame_i with Kalman filter
        predicted_frame_j = self.predictor.predict_objects(frame_i)

        # Match objects
        self.object_matcher.match_objects(predicted_frame_j, frame_j)
        #self.orb_feature_matcher(frame_i, frame_j)



        



