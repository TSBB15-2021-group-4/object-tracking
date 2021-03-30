#!/usr/bin/env python3
import numpy as np
from cv2 import cv2
import numpy as np
import math
from sklearn.preprocessing import normalize
from random import sample
from matplotlib import pyplot as plt
class GaussianMixtureModels():
    """
    Class representing the Gaussian mixture models approach for background modeling.
    """
    def __init__(self, alpha, w_init, variance_init, Threshold):
        """
        Parameters
        ----------
        variable_name : type
            Short description
        """
        # ...
        self.alpha = alpha
        self.w_init = w_init
        self.variance_init = variance_init
        self.Threshold = Threshold
        self.lambda_thr = 6.25
        self.K = 5

        self.blur_on = True
        self.blur_size = 5
        self.rgb_on = True
        self.gray_on = not(self.rgb_on)

    def compute_w_d_sigma(self, w, variance):
        return w / np.sqrt(variance)

    def sort(self, w_reshaped, mu_reshaped, variance_reshaped):
        w_d_sigma_reshaped = w_reshaped
        w_d_sigma_reshaped = np.where(variance_reshaped != 0, (w_reshaped / np.sqrt(variance_reshaped)),
                                      w_d_sigma_reshaped)
        # handling divide by zero and NAN values
        w_d_sigma_reshaped = np.where(np.logical_and(np.equal(variance_reshaped, 0), np.not_equal(w_reshaped, 0)), 200,
                                      w_d_sigma_reshaped)
        w_d_sigma_reshaped = np.where(np.logical_and(np.equal(variance_reshaped, 0), np.equal(w_reshaped, 0)), 0,
                                      w_d_sigma_reshaped)

        normal_dist = np.array([w_d_sigma_reshaped, w_reshaped, mu_reshaped, variance_reshaped])
        _ind = np.argsort(-normal_dist[0], axis=1)  # sort in a descending order
        ind = np.array([_ind, _ind, _ind, _ind])
        sorted_normal_dist = np.take_along_axis(normal_dist, ind, axis=2)
        w_reshaped = sorted_normal_dist[1]
        mu_reshaped = sorted_normal_dist[2]
        variance_reshaped = sorted_normal_dist[3]
        return w_reshaped, mu_reshaped, variance_reshaped

    def create_binary_images(self, frames, visualize):

        #for shadow detection
        alpha = 0.4
        beta  = 0.95
        Ts    = 0.1
        Th    = 0.1

        intensity_out_of_bound = 500
        tot_frames = len(frames)

        ten_per_gap = np.arange(0, tot_frames, tot_frames / 10)
        if self.gray_on:
            temp_f0 = frames[0].gray_image
            rows, cols = temp_f0.shape

            mu = np.zeros((rows, cols, self.K))
            mu += intensity_out_of_bound  # very large mean so that any pixel can be matched with this initial gaussian before it is created after match == 0
            variance = np.zeros((rows, cols, self.K))
            variance += 0.1  # to avoid divde by zero

            w = np.zeros([rows, cols, self.K])

            # the first gaussian will have its mean value equal to x_0 in frame at time 0
            mu[:, :, 0] = temp_f0
            # initial variance for the first gaussian: relatively large
            variance[:, :, 0] = self.variance_init *0.8
            # initial weight for the first gaussian: 1
            w[:, :, 0] = 1

            rho = np.zeros([rows, cols, self.K])

            self.alpha  # learning rate
            self.Threshold
            B_hat = np.ones([rows, cols, tot_frames])
            for t in range(tot_frames):
                if t in ten_per_gap:
                    temp = np.where(ten_per_gap == t)
                    print(temp[0][0] * 10, '% done')
                is_matched = np.zeros([rows, cols, self.K])
                w_d_sigma = np.zeros([rows, cols, self.K])
                w_d_sigma_only_is_match_true = np.zeros([rows, cols, self.K])
                m = np.zeros([rows, cols])
                match = np.zeros([rows, cols])

                f_t_gray = frames[t].gray_image

                # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                # f_t_gray = clahe.apply(f_t_gray)
                # equ = cv2.equalizeHist(f_t_gray)
                # f_t_gray = equ
                if self.blur_on:
                    f_t_gray = cv2.blur(f_t_gray,(self.blur_size, self.blur_size))
                # f_t_gray = cv2.medianBlur(f_t_gray, self.blur_size)

                f_t_exp_dim = np.expand_dims(f_t_gray, axis=2)
                dk_sq = np.zeros((rows, cols, self.K))
                dk_sq += 100  # initialize to have very large so that if there is no gaussian_k created, is_match returns false

                # dk_sq = np.power((f_t_4layers - mu), 2)
                # variance != 0 means there is k_th gaussian created, if its equal to zero, there is no k_th gaussian created yet
                with np.errstate(divide='ignore'):
                    dk_sq = np.where(np.not_equal(variance, 0), (f_t_exp_dim - mu) ** 2 / variance, dk_sq)
                is_matched = np.less_equal(dk_sq, self.lambda_thr)

                w_d_sigma = w / np.sqrt(variance)

                w_d_sigma_only_is_match_true = np.where(is_matched == False, 0, w_d_sigma)

                # 8, 9: else if wk_d_sigmak is greater, m = k
                m = np.argmax(w_d_sigma_only_is_match_true, axis=2)

                is_matched_cumsum = np.cumsum(is_matched, axis=2)  # 0 corresponds there is no such gaussian matched, ~0 means there are one or more gaussian matched

                match = is_matched_cumsum[:, :, -1]
                match = np.where(match < 1, match, 1)  # now it has either 0 / 1

                # 15: if match = 0 then

                # before line 15 algorithm 2

                # 16: m = K_temp
                # 19 march

                match_exp_dim = np.expand_dims(match, axis=2)

                if np.any(np.equal(match, 0)):
                    # m is where k_th gaussian hasn't been created
                    # which means where first mu == intensity_out_of_bound appears, where first variance == 0 appears, .. where first weight == 0 appears
                    m_temp = np.argmax(mu == intensity_out_of_bound, axis=2)  # shape; (rows, cols)
                    m = np.where(np.equal(match, 0), m_temp, m)
                    # if there is no such mu == intensity_out_of_bound, assign m = K - 1(which is 3)
                    m = np.where(np.logical_and(np.equal(match, 0), np.equal(m, 0)), (self.K - 1), m)
                    m_expended_dim = np.expand_dims(m, axis=2)

                    # 17 algorithm 2
                    w_temp_unmatched = np.copy(w)
                    np.put_along_axis(w_temp_unmatched, m_expended_dim, self.alpha, axis=2)
                    w = np.where(np.equal(match_exp_dim, 0), w_temp_unmatched, w)

                    # 18 algorithm 2
                    mu_temp_unmatched = np.copy(mu)
                    np.put_along_axis(mu_temp_unmatched, m_expended_dim, f_t_exp_dim, axis=2)
                    mu = np.where(np.equal(match_exp_dim, 0), mu_temp_unmatched, mu)

                    # 19
                    variance_temp_unmatched = np.copy(variance)
                    np.put_along_axis(variance_temp_unmatched, m_expended_dim, self.variance_init, axis=2)
                    variance = np.where(np.equal(match_exp_dim, 0), variance_temp_unmatched, variance)

                m_eye = np.eye(self.K)[m]
                # 20: else (match = 1)
                if np.any(np.equal(match, 1)):
                    # 21 algorithm 2
                    w_temp_matched = np.copy(w)
                    w_temp_matched = (1 - self.alpha) * w_temp_matched + self.alpha
                    w = np.where(np.logical_and(np.equal(match_exp_dim, 1), m_eye), w_temp_matched, w)

                    # 22
                    rho_temp_matched = np.copy(rho)
                    rho_temp_matched = self.alpha / w
                    rho = np.where(np.logical_and(np.equal(match_exp_dim, 1), m_eye), rho_temp_matched, rho)

                    # 23
                    mu_temp_matched = np.copy(mu)
                    mu_temp_matched = (1 - rho) * mu + rho * f_t_exp_dim
                    mu = np.where(np.logical_and(np.equal(match_exp_dim, 1), m_eye), mu_temp_matched, mu)

                    # 24
                    variance_temp_matched = np.copy(variance)
                    variance_temp_matched = (1 - rho) * variance + rho * (f_t_exp_dim - mu) * (f_t_exp_dim - mu)
                    variance = np.where(np.logical_and(np.equal(match_exp_dim, 1), m_eye), variance_temp_matched, variance)

                # 27 renormalize w
                # w_temp = w.reshape((rows * cols, self.K))
                # w_temp /= np.sum(w_temp, axis=1, keepdims=True)

                # algorithm 6 29 w_k = (1 - alpha) * w_k for all k's != m
                w_copied = np.copy(w)
                w_copied = w_copied*m_eye

                w = (1 - self.alpha) * w
                w = w * np.logical_not(m_eye) + w_copied

                # 29

                w_res = w.reshape((rows * cols, self.K))
                mu_res = mu.reshape((rows * cols, self.K))
                variance_res = variance.reshape((rows * cols, self.K))
                w_res_sorted, mu_res_sorted, variance_res_sorted = self.sort(w_res, mu_res, variance_res)
                w = w_res_sorted.reshape([rows, cols, self.K])
                mu = mu_res_sorted.reshape([rows, cols, self.K])
                variance = variance_res_sorted.reshape([rows, cols, self.K])

                # 33
                # get cumulative sum of w(weight)
                w_cumsum = np.cumsum(w, axis=2)

                B_mask = w_cumsum < self.Threshold
                first_false_arg = np.argmax(B_mask == False, axis=2)
                first_false_arg_exp_dims = np.expand_dims(first_false_arg, axis=2)
                np.put_along_axis(B_mask, first_false_arg_exp_dims, True, axis=2)
                B_mask = np.where(B_mask == False, intensity_out_of_bound, B_mask)
                d = (f_t_exp_dim - mu) ** 2 / variance  # compute the distances to all mixture components
                B_h_temp = np.any(np.less((B_mask * d), self.lambda_thr), axis=2)
                frames[t].binary_image = np.logical_not(B_h_temp).astype('uint8')  # use B as a mask to filter out non background components from the check

                #SHADOW DETECTION USING HSV values
                # Hypothesis is that a shadowed pixel value’s value and saturation
                # will decrease while the hue remains relatively constant.

                if(t==0):
                    pixels_states = frames[t].binary_image
                    previous_hhvvss = cv2.cvtColor(frames[t].rgb_image, cv2.COLOR_BGR2HSV)

                # H = channel 0 (Hue)
                # S = channel 1 (Saturation)
                # V = channel 2 (Lightness)

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

                Xbl = np.where(Xbl == 0, -1, Xbl)
                div  = np.divide(Xfl, Xbl)
                diff = np.array(Xfs) - np.array(Xbs)
                abs_diff = np.absolute(np.array(Xfh) - np.array(Xbh))

                cond1 = np.logical_and(alpha <= div , beta >= div)
                cond2 = diff <= Ts
                cond3 = abs_diff <= Th

                potential_shadowed_pixels = np.logical_and(cond1, cond2, cond3)
                potential_shadowed_pixels = np.logical_not(potential_shadowed_pixels)
                potential_shadowed_pixels = np.logical_and(potential_shadowed_pixels, moved_pixels)

                pixels_states = frames[t].binary_image
                #TODO shadow detection should be in a seperated function
                #frames[t].binary_image = np.where(potential_shadowed_pixels == 1, 0, frames[t].binary_image)

                previous_hhvvss = hsv_frame

                '''
                img = cv2.convertScaleAbs(frames[t].binary_image, alpha=(255.0))
                p = str(t) + ".png"
                cv2.imwrite(p, img)

                plt.imshow(frames[t].binary_image, cmap='gray')
                plt.pause(0.0001)
                plt.clf()
                '''

        if self.rgb_on:
            temp_f0 = frames[0].rgb_image
            rows, cols, channels = temp_f0.shape

            mu = np.zeros((rows, cols, channels, self.K))
            mu += intensity_out_of_bound  # very large mean so that any pixel can be matched with this initial gaussian before it is created after match == 0
            variance = np.zeros((rows, cols, channels, self.K))
            variance += 0.1  # to avoid divde by zero

            w = np.zeros([rows, cols, channels, self.K])

            # the first gaussian will have its mean value equal to x_0 in frame at time 0
            mu[:, :, :, 0] = temp_f0
            # initial variance for the first gaussian: relatively large
            variance[:, :, :, 0] = self.variance_init *0.7
            # initial weight for the first gaussian: 1
            w[:, :, :, 0] = 1

            rho = np.zeros([rows, cols, channels, self.K])

            self.alpha  # learning rate
            self.Threshold
            self.blur_size = 3
            B_hat = np.ones([rows, cols, tot_frames])
            for t in range(tot_frames):
                if t in ten_per_gap:
                    temp = np.where(ten_per_gap == t)
                    print(temp[0][0] * 10, '% done')
                is_matched = np.zeros([rows, cols, channels, self.K])
                w_d_sigma = np.zeros([rows, cols, channels, self.K])
                w_d_sigma_only_is_match_true = np.zeros([rows, cols, channels, self.K])
                m = np.zeros([rows, cols, channels])
                match = np.zeros([rows, cols, channels])

                f_t_col = frames[t].rgb_image

                # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                # f_t_gray = clahe.apply(f_t_gray)
                # equ = cv2.equalizeHist(f_t_gray)
                # f_t_gray = equ
                if self.blur_on:
                    f_t_col = cv2.blur(f_t_col,(self.blur_size, self.blur_size))
                # f_t_gray = cv2.medianBlur(f_t_gray, self.blur_size)

                f_t_exp_dim = np.expand_dims(f_t_col, axis=3)
                dk_sq = np.zeros((rows, cols, channels, self.K))
                dk_sq += 100  # initialize to have very large so that if there is no gaussian_k created, is_match returns false

                # dk_sq = np.power((f_t_4layers - mu), 2)
                # variance != 0 means there is k_th gaussian created, if its equal to zero, there is no k_th gaussian created yet
                with np.errstate(divide='ignore'):
                    dk_sq = np.where(np.not_equal(variance, 0), (f_t_exp_dim - mu) ** 2 / variance, dk_sq)
                is_matched = np.less_equal(dk_sq, self.lambda_thr)

                w_d_sigma = w / np.sqrt(variance)

                w_d_sigma_only_is_match_true = np.where(is_matched == False, 0, w_d_sigma)

                # 8, 9: else if wk_d_sigmak is greater, m = k
                m = np.argmax(w_d_sigma_only_is_match_true, axis=3)

                is_matched_cumsum = np.cumsum(is_matched, axis=3)  # 0 corresponds there is no such gaussian matched, ~0 means there are one or more gaussian matched

                match = is_matched_cumsum[:, :, :, -1]
                match = np.where(match < 1, match, 1)  # now it has either 0 / 1

                # 15: if match = 0 then

                # before line 15 algorithm 2

                # 16: m = K_temp
                # 19 march

                match_exp_dim = np.expand_dims(match, axis=3)

                if np.any(np.equal(match, 0)):
                    # m is where k_th gaussian hasn't been created
                    # which means where first mu == intensity_out_of_bound appears, where first variance == 0 appears, .. where first weight == 0 appears
                    m_temp = np.argmax(mu == intensity_out_of_bound, axis = 3)  # shape; (rows, cols)
                    m = np.where(np.equal(match, 0), m_temp, m)
                    # if there is no such mu == intensity_out_of_bound, assign m = K - 1(which is 3)
                    m = np.where(np.logical_and(np.equal(match, 0), np.equal(m, 0)), (self.K - 1), m)
                    m_expended_dim = np.expand_dims(m, axis=3)

                    # 17 algorithm 2
                    w_temp_unmatched = np.copy(w)
                    np.put_along_axis(w_temp_unmatched, m_expended_dim, self.alpha, axis=3)
                    w = np.where(np.equal(match_exp_dim, 0), w_temp_unmatched, w)

                    # 18 algorithm 2
                    mu_temp_unmatched = np.copy(mu)
                    np.put_along_axis(mu_temp_unmatched, m_expended_dim, f_t_exp_dim, axis=3)
                    mu = np.where(np.equal(match_exp_dim, 0), mu_temp_unmatched, mu)

                    # 19
                    variance_temp_unmatched = np.copy(variance)
                    np.put_along_axis(variance_temp_unmatched, m_expended_dim, self.variance_init, axis=3)
                    variance = np.where(np.equal(match_exp_dim, 0), variance_temp_unmatched, variance)

                m_eye = np.eye(self.K)[m]
                # 20: else (match = 1)
                if np.any(np.equal(match, 1)):
                    # 21 algorithm 2
                    w_temp_matched = np.copy(w)
                    w_temp_matched = (1 - self.alpha) * w_temp_matched + self.alpha
                    w = np.where(np.logical_and(np.equal(match_exp_dim, 1), m_eye), w_temp_matched, w)

                    # 22
                    rho_temp_matched = np.copy(rho)
                    rho_temp_matched = self.alpha / w
                    rho = np.where(np.logical_and(np.equal(match_exp_dim, 1), m_eye), rho_temp_matched, rho)

                    # 23
                    mu_temp_matched = np.copy(mu)
                    mu_temp_matched = (1 - rho) * mu + rho * f_t_exp_dim
                    mu = np.where(np.logical_and(np.equal(match_exp_dim, 1), m_eye), mu_temp_matched, mu)

                    # 24
                    variance_temp_matched = np.copy(variance)
                    variance_temp_matched = (1 - rho) * variance + rho * (f_t_exp_dim - mu) * (f_t_exp_dim - mu)
                    variance = np.where(np.logical_and(np.equal(match_exp_dim, 1), m_eye), variance_temp_matched, variance)

                # 27 renormalize w
                # w_temp = w.reshape((rows * cols, self.K))
                # w_temp /= np.sum(w_temp, axis=1, keepdims=True)

                # algorithm 6 29 w_k = (1 - alpha) * w_k for all k's != m
                w_copied = np.copy(w)
                w_copied = w_copied * m_eye

                w = (1 - self.alpha) * w
                w = w * np.logical_not(m_eye) + w_copied

                # 29

                w_res = w.reshape((rows * cols * channels, self.K))
                mu_res = mu.reshape((rows * cols * channels, self.K))
                variance_res = variance.reshape((rows * cols * channels, self.K))
                w_res_sorted, mu_res_sorted, variance_res_sorted = self.sort(w_res, mu_res, variance_res)
                w = w_res_sorted.reshape([rows, cols, channels, self.K])
                mu = mu_res_sorted.reshape([rows, cols, channels, self.K])
                variance = variance_res_sorted.reshape([rows, cols, channels, self.K])

                # 33
                # get cumulative sum of w(weight)
                w_cumsum = np.cumsum(w, axis=3)

                B_mask = w_cumsum < self.Threshold
                first_false_arg = np.argmax(B_mask == False, axis=3)
                first_false_arg_exp_dims = np.expand_dims(first_false_arg, axis=3)
                np.put_along_axis(B_mask, first_false_arg_exp_dims, True, axis=3)
                B_mask = np.where(B_mask == False, intensity_out_of_bound, B_mask)
                d = (f_t_exp_dim - mu) ** 2 / variance  # compute the distances to all mixture components
                B_h_temp = np.any(np.less((B_mask * d), self.lambda_thr), axis=3)
                B_h_temp = np.logical_not(B_h_temp)
                B_hat = np.logical_or(B_h_temp[..., 0], B_h_temp[..., 1], B_h_temp[..., 2])
                frames[t].binary_image = B_hat.astype('uint8')  # use B as a mask to filter out non background components from the check


                # SHADOW CORRECTION WITH HORPRASERT METHOD
                # algorithm 7
                # beta1 = 0.3
                # beta2 = 0.95
                # Tc = 0.04
                # mu_rgb = mu 

                # f_t_rgb = f_t_exp_dim
                # Dv_temp1 = f_t_rgb* mu # (x_r*mu_r, x_g*mu_g, x_b*mu_b)

                # norm = np.linalg.norm(mu, axis=2)
                # norm = np.where(norm == 0, -1, norm) #div by zero?
                # Dv = np.sum(Dv_temp1, axis=2) / norm
                # test = np.sum(Dv_temp1, axis=2)
                # test_temp = np.zeros((rows, cols, 3, self.K))
                # test_temp[:, :, 0, :] = test
                # test_temp[:, :, 1, :] = test
                # test_temp[:, :, 2, :] = test
                # Dc_temp = f_t_rgb - test_temp*mu
                # Dc = np.linalg.norm(Dc_temp, axis=2)

                # B_mask_shadow = B_mask[:, :, 0, :]

                # temp1 = np.any(np.greater_equal((B_mask_shadow * Dv), beta1), axis=2)
                # temp2 = np.any(np.less_equal((B_mask_shadow * Dv), beta2), axis=2)
                # temp3 = np.any(np.less_equal((B_mask_shadow * Dc), Tc), axis=2)
                # temp = np.logical_and(temp1, temp2, temp3)
                # temp = np.logical_and(frames[t].binary_image, temp)
                # frames[t].binary_image = np.where(temp == 1, 0, frames[t].binary_image)

                '''
                img = cv2.convertScaleAbs(frames[t].binary_image, alpha=(255.0))
                p = str(t) + ".png"
                cv2.imwrite(p, img)

                plt.imshow(frames[t].binary_image, cmap='gray')
                plt.pause(0.0001)
                plt.clf()
                '''

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
        pass





