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
    return w / np.sqrt(variance)


def sort(w_reshaped, mu_reshaped, variance_reshaped):
    w_d_sigma_reshaped = w_reshaped
    w_d_sigma_reshaped = np.where(variance_reshaped != 0, (w_reshaped / np.sqrt(variance_reshaped)), w_d_sigma_reshaped)
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

#TODO path should be inserted
frame_list = FrameList('C:\\Users\\rhk71\\Desktop\\TSBB15\\project1\\gitRepos\\tsbb15-project1-group4\\own\\MOT17-04-DPM-raw.webm')
frame_list.read_frames()

frames = frame_list.frames
tot_frames = frames.shape[0]

temp_f0 = cv2.cvtColor(frame_list.frames[0], cv2.COLOR_BGR2GRAY)
rows, cols = temp_f0.shape

img = cv2.convertScaleAbs(frame_list.frames[30])
p = str(30) + ".png"
cv2.imwrite(p, img)



# initial values setting
w_init = 0.1

variance_init = 10


K_max = 4 # max num of Gaussians(clusters)
K = 4 # how many gaussians have been created
lambda_thr = 2.5
lambda_thr = lambda_thr**2 # we will evaluate dk_square instead of dk

# initialize mu; randomly picked intensity from first frame
#mu = np.random.choice(temp_f0.flatten(), K * rows * cols).reshape((rows, cols, K_max))
#variance = np.random.rand(rows * cols * K).reshape((rows, cols, K_max))
#variance = variance + variance_init

mu = np.zeros((rows, cols, K_max))
mu += 300 # very large mean so that any pixel can be matched with this initial gaussian before it is created after match == 0
variance = np.zeros((rows, cols, K_max))
variance += 0.1 # to avoid divde by zero

w = np.zeros([rows, cols, K_max])

# the first gaussian will have its mean value equal to x_0 in frame at time 0
mu[:, :, 0] = temp_f0
# initial variance for the first gaussian: relatively large
variance[:, :, 0] = variance_init + 100
# initial weight for the first gaussian: 1
w[:, :, 0] = 1

rho = np.zeros([rows, cols, K_max])

alpha = 0.001 # learning rate
Threshold = 0.6

B = np.zeros([rows, cols])
B_hat = np.ones([rows, cols, tot_frames])
for t in range(tot_frames):
    print('Processing frame ', t)
    is_matched = np.zeros([rows, cols, K_max])
    w_d_sigma = np.zeros([rows, cols, K_max])
    w_d_sigma_only_is_match_true = np.zeros([rows, cols, K_max])
    m = np.zeros([rows, cols])
    match = np.zeros([rows, cols])

    f_t_gray = cv2.cvtColor(frame_list.frames[t], cv2.COLOR_BGR2GRAY)
    f_t_4layers = np.dstack((f_t_gray, f_t_gray, f_t_gray, f_t_gray))

    dk_sq = np.zeros((rows, cols, K_max))
    dk_sq += 100 # initialize to have very large so that if there is no gaussian_k created, is_match returns false

    #dk_sq = np.power((f_t_4layers - mu), 2)
    # variance != 0 means there is k_th gaussian created, if its equal to zero, there is no k_th gaussian created yet
    with np.errstate(divide='ignore'):
        dk_sq = np.where(np.not_equal(variance, 0), (f_t_4layers - mu)**2 / variance, dk_sq)


    is_matched = np.less_equal(dk_sq, lambda_thr)

    w_d_sigma = w / np.sqrt(variance)

    w_d_sigma_only_is_match_true = np.where(is_matched == False, 0, w_d_sigma)

    # 8, 9: else if wk_d_sigmak is greater, m = k
    m = np.argmax(w_d_sigma_only_is_match_true, axis=2)

    is_matched_cumsum = np.cumsum(is_matched, axis=2) # 0 corresponds there is no such gaussian matched, ~0 means there are one or more gaussian matched

    match = is_matched_cumsum[:, :, -1]
    match = np.where(match < 1, match, 1) # now it has either 0 / 1

    # 15: if match = 0 then

    # before line 15 algorithm 2
    # TODO only works when K = 4, has to be generalized
    f_t_gray_reshape = f_t_gray.reshape((rows * cols))
    f_t_gray_reshape_4cols = np.array([f_t_gray_reshape, f_t_gray_reshape, f_t_gray_reshape, f_t_gray_reshape],
                                      dtype=int).T
    # 16: m = K_temp
    ## m = np.where(match != 0, (K - 1), m)
    # 19 march

    match_exp_dim = np.expand_dims(match, axis=2)

    f_t_4cols_exp_dim = np.expand_dims(f_t_gray, axis=2)

    if np.any(np.equal(match, 0)):
        # m is where k_th gaussian hasn't been created
        # which means where first mu == 300 appears, where first variance == 0 appears, .. where first weight == 0 appears
        m_temp = np.argmax(mu == 300, axis=2)  # shape; (rows, cols)
        m = np.where(np.equal(match, 0), m_temp, m)
        # if there is no such mu == 300, assign m = K - 1(which is 3)
        m = np.where(np.logical_and(np.equal(match, 0), np.equal(m, 0)), (K-1), m)
        m_expended_dim = np.expand_dims(m, axis=2)

        # 17 algorithm 2
        w_temp_unmatched = np.copy(w)
        np.put_along_axis(w_temp_unmatched, m_expended_dim, w_init, axis = 2)
        w = np.where(np.equal(match_exp_dim, 0), w_temp_unmatched, w)

        # 18 algorithm 2
        mu_temp_unmatched = np.copy(mu)
        np.put_along_axis(mu_temp_unmatched, m_expended_dim, f_t_4cols_exp_dim, axis=2)
        mu = np.where(np.equal(match_exp_dim, 0), mu_temp_unmatched, mu)

        # 19
        variance_temp_unmatched = np.copy(variance)
        np.put_along_axis(variance_temp_unmatched, m_expended_dim, variance_init, axis = 2)
        variance = np.where(np.equal(match_exp_dim, 0), variance_temp_unmatched, variance)

    m_eye = np.eye(4)[m]
    # 20: else (match = 1)
    if np.any(np.equal(match, 1)):
        # 21 algorithm 2
        w_temp_matched = np.copy(w)
        w_temp_matched = (1 - alpha) * w_temp_matched + alpha
        w = np.where(np.logical_and(np.equal(match_exp_dim, 1), m_eye), w_temp_matched, w)

        # 22
        rho_temp_matched = np.copy(rho)
        rho_temp_matched = alpha / w
        rho = np.where(np.logical_and(np.equal(match_exp_dim, 1), m_eye), rho_temp_matched, rho)

        # 23
        mu_temp_matched = np.copy(mu)
        mu_temp_matched = (1 - rho) * mu + rho * f_t_4cols_exp_dim
        mu = np.where(np.logical_and(np.equal(match_exp_dim, 1), m_eye), mu_temp_matched, mu)

        # 24
        variance_temp_matched = np.copy(variance)
        variance_temp_matched = (1 - rho) * variance + rho * (f_t_4cols_exp_dim - mu) * (f_t_4cols_exp_dim - mu)
        variance = np.where(np.logical_and(np.equal(match_exp_dim, 1), m_eye), variance_temp_matched, variance)

    # 27 renormalize w
    w_temp = w.reshape((rows * cols, K))
    w_temp /= np.sum(w_temp, axis=1, keepdims=True)

    # 29

    w_res = w.reshape((rows * cols, K))
    mu_res = mu.reshape((rows * cols, K))
    variance_res = variance.reshape((rows * cols, K))
    w_res_sorted, mu_res_sorted, variance_res_sorted = sort(w_res, mu_res, variance_res)
    w = w_res_sorted.reshape([rows, cols, K])
    mu = mu_res_sorted.reshape([rows, cols, K])
    variance = variance_res_sorted.reshape([rows, cols, K])

    # 33
    # get cumulative sum of w(weight)
    w_cumsum = np.cumsum(w, axis=2)

    B_mask = w_cumsum < Threshold
    first_false_arg = np.argmax(B_mask == False, axis=2)
    first_false_arg_exp_dims = np.expand_dims(first_false_arg, axis=2)
    np.put_along_axis(B_mask, first_false_arg_exp_dims, True, axis=2)
    B_mask = np.where(B_mask == False, 300, B_mask)
    d = (f_t_4layers - mu) ** 2 / variance # compute the distances to all mixture components
    B_h_temp = np.any(np.less((B_mask * d), lambda_thr), axis=2)
    B_hat[:, :, t] = np.logical_not(B_h_temp)  # use B as a mask to filter out non background components from the check

    plt.imshow(B_hat[:, :, t], cmap='gray')
    plt.pause(0.0001)
    plt.clf()


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



