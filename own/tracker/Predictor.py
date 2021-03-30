#!/usr/bin/env python3

import numpy as np
import copy

class Predictor():
    """
    Class representing the Kalman filter used to predict objects positions.
    """

    def __init__(self, Q0, R0, P0, T):
        """
        Parameters
        ----------
        Q0 : float
            Initial covariance of process noise
        R0 : float
            Initial covariance to measurement noise
        P0 : numpy array 1x4
            Covariance for initial state
        T  : float
            Sampling time
        """

        # Setup the motion model: 
        # | x[t+1] = A*x[t] + B*w[t]
        # | y[t]   = C*x[t] + v[t]
        self.A = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.B = np.array([[T ** 2 / 2, 0], [0, T ** 2 / 2], [T, 0], [0, T]])
        self.C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Covariance of process noise w[t]
        self.Q = self.B @ np.identity(2) @ self.B.T * Q0

        # Covariance of measurement noise v[t]
        self.R = np.identity(2) * R0

        # Covariance
        self.P = np.diag(P0)

        # Kalman gain
        self.K = None 


    def update_kalman_gain(self):
        self.K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.R)


    def update_cov_estimate(self):
        self.P = self.P - self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.R) @ self.C @ self.P
        self.P = self.A @ self.P @ self.A.T + self.Q
    

    def predict_object(self, obj):
        """Predict next state of object obj."""
        # Measurement update
        obj.state = obj.predicted_state + self.K @ (np.array([[obj.pos_x], [obj.pos_y]]) - self.C @ obj.predicted_state)

        # Predict next position
        obj.predicted_state = (self.A - self.A @ self.K @ self.C) @ obj.state + \
                              self.A @ self.K @ (self.C @ obj.state)

        # Update the pos_y and pos_x to predicted ones
        obj.pos_x = obj.predicted_state[0]
        obj.pos_y = obj.predicted_state[1]

    def predict_objects(self, frame):
        """Predicts the state of all objects present in frame."""      

        predicted_frame = copy.deepcopy(frame)
        self.update_kalman_gain()

        for obj in predicted_frame.object_list:
            self.predict_object(obj)

        self.update_cov_estimate()
        
        return predicted_frame