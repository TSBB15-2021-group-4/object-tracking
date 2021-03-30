#!/usr/bin/env python3

import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
import random



class MedianFiltering():
    """
    Class representing the median filtering approach for background modeling.
    """
    def __init__(self, alpha, T):
        """
        Parameters
        ----------
        alpha : float
            The adaption rate for the median filter

        T : int
            The thresholding value for the binary image
        """
        self.alpha = alpha
        self.T = T

    def create_binary_images(self, frames, visualize):
        """Given all the frames, set all frames' likelihood image"""
        # approx median,
        # converge slowly to background
        # m = np.ones((height, width))
        # m *= 128
        # or assume first frame is background
        m = frames[0].rgb_image
        tot_frames = len(frames)

        for t in range(tot_frames):
            # What the code does:
            # for i in range(width):
            #     for j in range(height):
            #         if frame_list.frames[t][j][i] > m[j][i]:
            #             m[j][i] += alpha
            #         else:
            #             m[j][i] -= alpha
            #
            #         # segment binary
            #         if abs(frame_list.frames[t][j][i] - m[j][i]) > T:
            #             binary[t][j][i] = 1
            # Vectorized
            pix_largthan_med = np.greater(frames[t].rgb_image, m)
            m = m + self.alpha * pix_largthan_med + self.alpha * (pix_largthan_med - 1)  # +alpha where it's larger, -alpha where smaller
            bin_im_rgb = 1*np.greater(np.absolute(frames[t].rgb_image - m), self.T).astype('uint8')
            frames[t].binary_image = np.zeros((bin_im_rgb.shape[0],bin_im_rgb.shape[1]))
            for i in range(2):
                frames[t].binary_image = np.logical_or(frames[t].binary_image, bin_im_rgb[:,:,i])
            frames[t].binary_image = frames[t].binary_image.astype('uint8')

        # To visualize every 5th frame:
        if visualize:
            self.visualize_filtering(frames)
        

    def visualize_filtering(self, frames):
        tot_frames = len(frames)
        plt.figure()
        i = 0
        display_image = True
        while display_image:
            plt.imshow(frames[i].binary_image, cmap='gray')
            plt.pause(0.0001)
            plt.clf()
            i += 5
            if i >= tot_frames:
                display_image = False


    def suppress_shadows(self, frames, visualize):
        # Hypothesis is that a shadowed pixel value’s value and saturation 
        # will decrease while the hue remains relatively constant.
        
        # H = channel 0 (Hue)
        # S = channel 1 (Saturation)
        # V = channel 2 (Lightness)

        alpha = 0.3
        beta  = 0.9
        Ts    = 0.3
        Th    = 0.5

        tot_frames = len(frames)
        pixels_states = frames[0].binary_image
        previous_hhvvss = cv2.cvtColor(frames[0].rgb_image, cv2.COLOR_BGR2HSV)
        for t in range(tot_frames):
            print("frame no: ", t)
    
            hsv_frame = cv2.cvtColor(frames[t].rgb_image, cv2.COLOR_BGR2HSV)

            temp1 = np.logical_xor(frames[t].binary_image, pixels_states)
            temp2 = frames[t].binary_image # foreground == 1

            moved_pixels = np.logical_and(temp1, temp2)

            
            Xfl = hsv_frame[:, :, 2]
            Xfs = hsv_frame[:, :, 1]
            Xfh = hsv_frame[:, :, 0]

            Xbl = previous_hhvvss[:, :, 2]
            Xbs = previous_hhvvss[:, :, 1]
            Xbh = previous_hhvvss[:, :, 0]

            Xbl = np.where(Xbl <= 0, -1, Xbl)
            div  = np.divide(Xfl, Xbl)
            diff = np.array(Xfs) - np.array(Xbs)
            abs_diff = np.absolute(np.array(Xfh) - np.array(Xbh))

            cond1 = np.logical_and(alpha <= div , beta >= div)
            cond2 = diff <= Ts
            cond3 = abs_diff <= Th

            potential_shadowed_pixels = np.logical_and(cond1, cond2, cond3)
            potential_shadowed_pixels = np.logical_not(potential_shadowed_pixels)
            potential_shadowed_pixels = np.logical_and(potential_shadowed_pixels, moved_pixels)

            frames[t].binary_image = np.where(potential_shadowed_pixels == 1, 0.25, frames[t].binary_image)

            pixels_states = frames[t].binary_image
            previous_hhvvss = hsv_frame

        if visualize:
            self.visualize_filtering(frames)

            # width  = p.shape[1]
            # height = p.shape[0]
            # for i in range(width):
            #     for j in range(height):
            #         if (frames[t].binary_image[j][i] == 1  # loop over all foreground pixels
            #             and frames[t].binary_image[j][i] != pixels_states[j][i]):  # which have changed

            #             Xfl = hsv_frame[j][i][2]

            #             Xbl = self.calculate_previous_hsv_values(previous_hhvvss, j, i, 2)
            #             if(Xbl == 0):
            #                 break

            #             Xfs = hsv_frame[j][i][1]
            #             Xbs = self.calculate_previous_hsv_values(previous_hhvvss, j, i, 1)

            #             Xfh = hsv_frame[j][i][0]
            #             Xbh = self.calculate_previous_hsv_values(previous_hhvvss, j, i, 0)

            #             if( (alpha <= (Xfl/Xbl) <= beta) 
            #                 and ((Xfs - Xbs) <= Ts) 
            #                 and (abs(Xfh - Xbh) <= Th) ):
            #                     #print("pixel ", j, i, "unaltered")
            #                     frames[t].binary_image[j][i] = 1
            #             else:
            #                     #print("pixel ", j, i, "altered")
            #                     frames[t].binary_image[j][i] = 0.5

