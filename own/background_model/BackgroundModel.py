#!/usr/bin/env python3
from background_model.GaussianMixtureModels import GaussianMixtureModels
from background_model.MedianFiltering import MedianFiltering


class BackgroundModel():
    """
    Class with functions responsible for creating binary images representing
    background and foreground, and also for doing shadow suppresion.
    """

    def __init__(self, filter_type, m_alpha=1, m_T=50, g_alpha = 0.01, g_w_init = 0.001, g_variance_init= 200, g_T = 0.7):
        # MOT17-02 gaussian parameters; g_alpha = 0.01, g_w_init = 0.01, g_variance_init= 200, g_T = 0.7
        """
        Parameters
        ----------
        filter_type : str
            'median' for MedianFiltering or 'gaussian' for GaussianMixtureModels
        filter : MedianFiltering or GaussianMixtureModels
            Object of different approaches of background modeling

        alpha : float
            The adaption rate for the median filter
        T : int
            The thresholding value for the binary image
        """
        self.binary_image = []
        self.filter_type = filter_type
        self.m_alpha = m_alpha
        self.m_T = m_T
        self.g_alpha = g_alpha
        self.g_w_init = g_w_init
        self.g_variance_init = g_variance_init
        self.g_T = g_T

        if self.filter_type == 'median':
            self.filter = MedianFiltering(self.m_alpha, self.m_T)
        elif self.filter_type == 'gaussian':
            self.filter = GaussianMixtureModels(self.g_alpha, self.g_w_init, self.g_variance_init, self.g_T)

    def create_binary_images(self, frames, visualize=False):
        """Given all the frames, set all frames' likelihood image"""
        self.binary_image = self.filter.create_binary_images(frames, visualize)

    def suppress_shadows(self, frames, visualize=False):
        """Given all frames, set each frame's likelihood image"""
        self.filter.suppress_shadows(frames,visualize)
