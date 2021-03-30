import cv2
import numpy as np
import math
from sklearn.preprocessing import normalize
from random import sample
from matplotlib import pyplot as plt


class FrameList():
    """
    Class to represent a list of frames (image sequence).
    """

    def __init__(self, video_path):
        """
        Parameters
        ----------
        frames : [Frame, Frame, ...]
            1D array of frames
        video_path : str
            String containing the relative path to the video sequence
        """

        self.frames = []
        self.video_path = video_path

    def read_frames(self):
        """Read data from video sequence, build self.frames"""

        cap = cv2.VideoCapture(self.video_path)  # input, video file

        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        temp_frames = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
        fc = 0
        ret = True
        while (fc < frameCount and ret):
            ret, temp_frames[fc] = cap.read()
            fc += 1

        # for each image:
        #   new_frame = Frame(number=fc, original_image=image)
        #   self.frames.append(new_frame)

        # release some amount of memory?
        cap.release()

        self.frames = temp_frames


def compute_w_d_sigma(w, variance):
    var_abs = np.abs(variance)
    return w / np.sqrt(var_abs)


def sort(w_reshaped, mu_reshaped, variance_reshaped):
    w_d_sigma_reshaped = compute_w_d_sigma(w_reshaped, variance_reshaped)
    normal_dist = np.array([w_d_sigma_reshaped, w_reshaped, mu_reshaped, variance_reshaped])
    _ind = np.argsort(-normal_dist[0], axis=1)
    ind = np.array([_ind, _ind, _ind, _ind])
    sorted_normal_dist = np.take_along_axis(normal_dist, ind, axis=2)
    w_reshaped = sorted_normal_dist[1]
    mu_reshaped = sorted_normal_dist[2]
    variance_reshaped = sorted_normal_dist[3]
    return w_reshaped, mu_reshaped, variance_reshaped

def sort_test(w_reshaped, mu_reshaped, variance_reshaped):
    w_d_sigma_reshaped = compute_w_d_sigma(w_reshaped, variance_reshaped)
    normal_dist = np.array([w_d_sigma_reshaped, w_reshaped, mu_reshaped, variance_reshaped])
    _ind = np.argsort(-normal_dist[0], axis=1)
    ind = np.array([_ind, _ind, _ind, _ind])
    sorted_normal_dist = np.take_along_axis(normal_dist, ind, axis=2)
    w_reshaped = sorted_normal_dist[1]
    mu_reshaped = sorted_normal_dist[2]
    variance_reshaped = sorted_normal_dist[3]
    return w_reshaped, mu_reshaped, variance_reshaped


frame_list = FrameList(
    'C:\\Users\\rhk71\\Desktop\\TSBB15\\project1\\gitRepos\\tsbb15-project1-group4\\own\\test_data\\PETS09-S2L1-raw.mp4')
frame_list.read_frames()

'''
class for gaussians for each pixel
class normal_distribution:
  def __init__(self, mu, sigma, w, w_d_sigma):
    self.mu = mu
    self.sigma = sigma
    self.w = w
    self.w_d_sigma = w_d_sigma

  def mu(self):
    print("mu is " + self.mu)'''

m = 0
given_iter = 1
frames = frame_list.frames
tot_frames = frames.shape[0]
rows = frames.shape[1]
cols = frames.shape[2]
B_gray = np.empty((tot_frames, rows, cols))
B_RGB = np.empty((tot_frames, rows, cols, 3))

temp_f0 = cv2.cvtColor(frame_list.frames[0], cv2.COLOR_BGR2GRAY)
rows, cols = temp_f0.shape

# initial values setting
w_init = 2

variance_init = 2
variance_min = 0.001

K_max = 4  # num of Gaussians(clusters)?
K = 4
lambda_thr = 2.5
lambda_thr = pow(lambda_thr, 2)
#mu = np.dstack((temp_f0, temp_f0, temp_f0, temp_f0))
#mu = np.zeros([rows, cols])

mu = np.random.choice(temp_f0.flatten(), K*rows*cols).reshape((rows, cols, K))
variance = np.random.rand(rows * cols * K).reshape((rows, cols, K))
#variance = variance + variance_init

mu[:, :, 0] = temp_f0 # initial mu: xt
variance[:, :, 0] += 3 # initial variance : relatively large
rho = np.zeros([rows, cols, K])
# w should be normalized
#w = np.random.rand(K * rows * cols).reshape((rows * cols, K))
# instead of giving weight of 1 to N[x_t, ..], give relatively large weight
#w = w.reshape((rows, cols, K))
#w[:, :, 0] +=  3# here give relatively large to first gaussian
#w = w.reshape((rows*cols, K))
#w = normalize(w, axis=1, norm='l1')
#sort_idx = np.argsort(-w, axis=1)
#w = np.take_along_axis(w, sort_idx, axis = 1)
#w = w.reshape((rows, cols, K))
w = np.zeros([rows, cols, K])
w[:, :, 0] =  1
'''
#sort respect to (w/sigma)
w, mu, variance = sort(w, mu, variance)
'''

alpha = 0.01
Threshold = 0.7
match = np.zeros([rows, cols])
m = np.zeros([rows, cols])
B = np.zeros([rows, cols])
B_hat = np.zeros([rows, cols, tot_frames])
for t in range(0, tot_frames):
    print('Processing frame ', t)
    for k in range(0, K):
        f_t_gray = cv2.cvtColor(frame_list.frames[t], cv2.COLOR_BGR2GRAY)
        dk_sq = np.power((f_t_gray - mu[..., k]), 2)
        dk_sq = dk_sq / variance[..., k]
        #dk = np.sqrt(dk_sq)
        is_matched = np.less(dk_sq, lambda_thr)

        # matched
        '''if match == 0: # if it was 'matched' in the previous stage?
            m = k'''
        # TODO test this part

        target_pixels = np.logical_and(is_matched, np.logical_not(match))  # ~match: 0 becomes true
        target_pixels_to_zeros = m * np.logical_not(target_pixels)  # only target pixels become zero
        m = target_pixels_to_zeros + k * target_pixels  # now add k to target pixels which are zero

        temp1 = m.reshape((rows * cols))
        temp2 = np.arange(rows * cols)
        temp3 = np.array([temp2, temp1], dtype=int).T
        temp4 = temp3.tolist()
        temp_rows_cols, temp_m = zip(*temp4)

        w_temp = w.reshape((rows * cols, K))
        w_m = w_temp[temp_rows_cols, temp_m].reshape((rows, cols))

        var_temp = variance.reshape((rows * cols, K))
        variance_m = var_temp[temp_rows_cols, temp_m].reshape((rows, cols))

        target_pixels = np.logical_and(is_matched, match)  # match should be 1
        test = compute_w_d_sigma(w_m, variance_m)
        is_wk_d_sigmak_greater = np.greater(compute_w_d_sigma(w[..., k], variance[..., k]),
                                            compute_w_d_sigma(w_m, variance_m))

        target_pixels = np.logical_and(target_pixels, is_wk_d_sigmak_greater)

        # logical not -> true pixels become 0,
        # by multiplying (negation of target_pixels) with m, only target pixels become zero
        target_pixels_to_zeros = m * np.logical_not(target_pixels)
        m = target_pixels_to_zeros + k * target_pixels  # now add k to target pixels which are zero

        # match = 1
        # matched with one of existing gaussians, adjust mu and sigma
        target_pixels_to_zeros = match * np.logical_not(is_matched)
        match = target_pixels_to_zeros + is_matched  # target pixels get assigned 1(TRUE)

        '''
        if match == 0:
            # not matched with any existing gaussians,
            # substitute the last gaussian with a new gaussian

            m = K - 1

            w[m] = w_init
            normalize(w)

            mu[m] = x_t
            sigma[m] = sigma_init
        else: # adjust the existing gaussian's mu and sigma
            w[m] = (1 - alpha)*w[m] + alpha
            normalize(w)

            rho[m] = alpha / w[m]
            mu[m] = (1-rho[m])*mu[m] + rho[m]*x_t
            sigma[m] = (1.0 - rho[m]) * pow(sigma[m], 2) + rho[m] * ((x_t - mu[m]) * (x_t - mu[m]))
            sigma[m] = math.sqrt(sigma[m])
            if sigma[m] < sigma_min:
                sigma[m] = sigma_min * 1.01
        '''
    # 15: if match = 0 then
    #target_pixels = np.logical_not(match)  # ~match: 0 becomes true

    # 16: m = K
    #target_pixels_to_zeros = m * np.logical_not(target_pixels)  # only target pixels become zero
    #m1 = target_pixels_to_zeros + (K - 1) * target_pixels  # now add k to target pixels which are zero

    #TODO march 7 fix here m=K line 15
    m = np.where(match == 0, 1, m)
    temp_match = match.reshape((rows * cols))
    match_rows_cols = np.where(temp_match == 0)  # exclude match == 1

    #TODO only works when K = 4, has to be generalized
    f_t_gray_reshape = f_t_gray.reshape((rows * cols))
    f_t_gray_reshape_4cols = np.array([f_t_gray_reshape, f_t_gray_reshape, f_t_gray_reshape, f_t_gray_reshape],
                                      dtype=int).T

    # create a new gaussian by setting up wm, mum, sigmam
    temp1 = m.reshape((rows * cols))
    m_match0 = temp1[match_rows_cols]  # all values are supposed to be (K-1)
    temp3 = np.array([match_rows_cols[0], m_match0], dtype=int).T
    if (temp3.shape[0]):  # match == 0 pixels exists
        temp4 = temp3.tolist()
        temp_rows_cols, temp_m = zip(*temp4)

        # 17: update w[..., m]
        w_temp = w.reshape((rows * cols, K))
        w_temp[temp_rows_cols, temp_m] = w_init
        w = w_temp.reshape((rows, cols, K))
        # 18
        mu_temp = mu.reshape((rows * cols, K))
        mu_temp[temp_rows_cols, temp_m] = f_t_gray_reshape_4cols[temp_rows_cols, temp_m]
        mu = mu_temp.reshape((rows, cols, K))
        # 19
        variance_temp = w.reshape((rows * cols, K))
        variance_temp[temp_rows_cols, temp_m] = variance_init
        variance = variance_temp.reshape((rows, cols, K))

    # 20: else (match = 1)
    temp_match = match.reshape((rows * cols))
    match_rows_cols = np.where(temp_match == 1)  # exclude match == 1

    temp1 = m.reshape((rows * cols))
    m_match1 = temp1[match_rows_cols]
    temp3 = np.array([match_rows_cols[0], m_match1], dtype=int).T
    temp4 = temp3.tolist()
    temp_rows_cols, temp_m = zip(*temp4)

    w_temp = w.reshape((rows * cols, K))
    w_temp[temp_rows_cols, temp_m] = (1 - alpha) * w_temp[temp_rows_cols, temp_m] + alpha
    w = w_temp.reshape((rows, cols, K))

    rho_temp = rho.reshape((rows * cols, K))
    rho_temp[temp_rows_cols, temp_m] = alpha / w_temp[temp_rows_cols, temp_m]
    rho = rho_temp.reshape((rows, cols, K))

    mu_temp = mu.reshape((rows * cols, K))
    temp = (1 - rho_temp[temp_rows_cols, temp_m]) * mu_temp[temp_rows_cols, temp_m] + rho_temp[temp_rows_cols, temp_m] * \
           f_t_gray_reshape_4cols[temp_rows_cols, temp_m]
    mu_temp[temp_rows_cols, temp_m] = temp
    mu = mu_temp.reshape((rows, cols, K))

    variance_temp = w.reshape((rows * cols, K))
    temp = (1 - rho_temp[temp_rows_cols, temp_m]) * variance_temp[temp_rows_cols, temp_m] + rho_temp[
        temp_rows_cols, temp_m] * (
                       (f_t_gray_reshape_4cols[temp_rows_cols, temp_m] - mu_temp[temp_rows_cols, temp_m]) * (
                           f_t_gray_reshape_4cols[temp_rows_cols, temp_m] - mu_temp[temp_rows_cols, temp_m]))
    variance_temp[temp_rows_cols, temp_m] = temp
    variance = variance_temp.reshape((rows, cols, K))

    # 27 renormalize w
    w_temp = w.reshape((rows * cols, K))
    w_temp = normalize(w_temp, axis=1, norm='l1')
    w = w_temp.reshape(rows, cols, K)

    # 29
    temp_match = match.reshape((rows * cols))
    match_rows_cols = np.where(temp_match == 1)[0]

    w_temp = w.reshape((rows * cols, K))
    w_matched = w_temp[match_rows_cols, :]

    mu_temp = mu.reshape((rows * cols, K))
    mu_matched = mu_temp[match_rows_cols, :]

    variance_temp = variance.reshape((rows * cols, K))
    variance_matched = variance_temp[match_rows_cols, :]
    '''test1 = np.reciprocal(variance_un_matched)
    test = w_un_matched*test1'''
    w_temp, mu_temp, variance_temp = sort(w_temp, mu_temp, variance_temp)
    #w_matched, mu_matched, variance_matched = sort(w_matched, mu_matched, variance_matched)
    #w_temp, mu_temp, variance_temp = sort_test(w_temp, mu_temp, variance_temp)
    #w_temp2 = w_temp
    #mu_temp2 = mu_temp
    #variance_temp2 = variance_temp
    '''
    w_temp[temp_rows_cols, :] = w_matched[:, :]
    mu_temp[temp_rows_cols, :] = mu_matched[:, :]
    variance_temp[temp_rows_cols, :] = variance_matched[:, :]
    '''
    '''
    # test = np.put_along_axis(w_temp, match_rows_cols, w_matched, axis=0)
    for i in range(0, len(match_rows_cols)):
        w_temp[match_rows_cols[i], :] = w_matched[i, :]
        mu_temp[match_rows_cols[i], :] = mu_matched[i, :]
        variance_temp[match_rows_cols[i], :] = variance_matched[i, :]
    '''
    w = w_temp.reshape(rows, cols, K)
    mu = mu_temp.reshape(rows, cols, K)
    variance = variance.reshape(rows, cols, K)

    # 33
    # get cumulative sum of w(weight)
    w_cumsum = np.cumsum(w, axis=2)

    B = np.apply_along_axis(np.searchsorted, 2, w_cumsum, Threshold)

    # algorithm 3

    # line 2
    B_hat  # shape [rows, cols, tot_frames], segmantized binary img, initialized to be zero
    # TODO has to be generalized for different K than 4
    f_t_K_dim = np.dstack((f_t_gray, f_t_gray, f_t_gray, f_t_gray))
    dk_sq = np.power((f_t_K_dim - mu), 2)
    dk_sq = dk_sq / variance
    #dk = np.sqrt(dk_sq)

    cond = np.less(dk_sq, lambda_thr)
    cond_res = cond.reshape([rows * cols, K])
    cond_cumsum = np.cumsum(cond_res, axis=1)
    cond_cumsum = np.greater_equal(cond_cumsum, 1)

    B_res = B.reshape([rows * cols])
    temp_rows_cols = np.arange(rows*cols)

    temp = np.array([temp_rows_cols, B_res], dtype=int).T
    temp_list = temp.tolist()
    temp_rows_cols, temp_B = zip(*temp_list)

    temp = cond_cumsum[temp_rows_cols, temp_B].reshape([rows, cols])
    B_hat[:, :, t] = temp

    # shadow detection
    # algorithm 7
    # line 11


    if t == 3:
        plt.figure()
        plt.imshow(B_hat[:, :, t], cmap='gray')
        plt.show()

    img = cv2.convertScaleAbs(B_hat[:, :, t], alpha=(255.0))
    p = str(t) + ".png"
    cv2.imwrite(p, img)

''' FOR SHADOW
    locs = np.where(variance < variance_min)
    variance[locs] = variance_min * 1.01

    temp1 = m.reshape((rows * cols))
    temp2 = np.arange(rows * cols)
    temp3 = np.array([temp2, temp1], dtype=int).T
    temp4 = temp3.tolist()
    temp_rows_cols, temp_m = zip(*temp4)

    w_vec = w.reshape((rows * cols, K))
    w_m_vec = w_vec[temp_rows_cols, temp_m]


    for k in range(0, K):
        w[..., k] = (1 - alpha)*w[..., k]

    w.reshape((rows * cols, K))[temp_rows_cols, temp_m] = w_vec[temp_rows_cols, temp_m]
'''



