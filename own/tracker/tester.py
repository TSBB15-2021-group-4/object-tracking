#!/usr/bin/env python3

import numpy as np
import cv2
from matplotlib import pyplot as plt

import sys
sys.path.append('own') 
from ObjectMatcher import ObjectMatcher
#from data_reading.Frame import Frame
#from foreground_segmentation.Object import Object


def test_object_matcher():
    object_matcher = ObjectMatcher(match_algorithm='simple')

    # ================= Test creation of frames and adding of objects ===============

    frame_1 = Frame(1, np.ones((50, 50)))
    frame_2 = Frame(2, np.ones((50, 50)))
    
    # ------ Assert that the overlap between two 10x10 objects on the same position is 1010=100
    frame_1.object_list.append(Object(obj_id=1, box_width=10, box_height=10, pos_x=5, pos_y=5, center_x=10, center_y=10))
    frame_2.add_object(Object(obj_id=1, box_width=10, box_height=10, pos_x=5, pos_y=5, center_x=10, center_y=10))
    assert(object_matcher.overlap_score(frame_1.object_list[0], frame_2.object_list[0]) == 100)

    # ------ Assert that frames have proper sizes when adding objects
    assert(len(frame_1.object_list) == 1)
    assert(len(frame_2.object_list) == 1)

    # ------ Assert not the same object is added to the different frames
    assert(frame_1.object_list[0] != frame_2.object_list[0])

    # ======================= Test object_matcher.overlap_score() ==================

    # ------ Assert that overlap between two non-overlapping objects is 0
    frame_1.add_object(Object(obj_id=2, box_width=5, box_height=10, pos_x=30, pos_y=21, center_x=32.5, center_y=26))    
    assert(object_matcher.overlap_score(frame_1.object_list[1], frame_2.object_list[0]) == 0)

    # ------ Assert some more overlaps
    score = object_matcher.overlap_score(\
                    Object(obj_id=1, box_width=4, box_height=4, pos_x=0, pos_y=0, center_x=1, center_y=0), \
                    Object(obj_id=1, box_width=2, box_height=4, pos_x=0, pos_y=0, center_x=1, center_y=0))
    assert(score == 2*4) 
    score = object_matcher.overlap_score(\
                    Object(obj_id=1, box_width=4, box_height=4, pos_x=0, pos_y=0, center_x=1, center_y=0), \
                    Object(obj_id=1, box_width=1, box_height=4, pos_x=0, pos_y=0, center_x=1, center_y=0))
    assert(score == 1*4) 
    score = object_matcher.overlap_score(\
                    Object(obj_id=1, box_width=4, box_height=4, pos_x=0, pos_y=0, center_x=1, center_y=0), \
                    Object(obj_id=1, box_width=4, box_height=4, pos_x=2, pos_y=2, center_x=1, center_y=0))
    assert(score == 2*2) 

    # ==================== Test object_matcher.match_objects() =====================
    frame_1 = Frame(1, np.ones((50, 50)))
    frame_2 = Frame(2, np.ones((50, 50)))

    frame_1.add_object(Object(obj_id=1, box_width=4, box_height=4, pos_x=0, pos_y=0, center_x=1, center_y=0))
    frame_1.add_object(Object(obj_id=2, box_width=4, box_height=4, pos_x=10, pos_y=10, center_x=1, center_y=0))
    frame_1.add_object(Object(obj_id=3, box_width=4, box_height=4, pos_x=20, pos_y=20, center_x=1, center_y=0))
    frame_1.add_object(Object(obj_id=4, box_width=4, box_height=4, pos_x=30, pos_y=30, center_x=1, center_y=0))
    
    obj_1 = Object(box_width=4, box_height=4, pos_x=2, pos_y=1, center_x=1, center_y=0)
    obj_2 = Object(box_width=4, box_height=4, pos_x=12, pos_y=11, center_x=1, center_y=0)
    obj_3 = Object(box_width=4, box_height=4, pos_x=22, pos_y=21, center_x=1, center_y=0)
    obj_4 = Object(box_width=4, box_height=4, pos_x=32, pos_y=31, center_x=1, center_y=0)

    frame_2.add_object(obj_1)
    frame_2.add_object(obj_2)
    frame_2.add_object(obj_3)
    frame_2.add_object(obj_4)

    object_matcher.match_objects(frame_1, frame_2)

    assert(frame_2.object_list[0].id == 1)
    assert(frame_2.object_list[1].id == 2)
    assert(frame_2.object_list[2].id == 3)
    assert(frame_2.object_list[3].id == 4)

def test_feature_detector():
    img = cv2.imread('frame1.jpg')
    print(img.shape)

    # Initiate STAR detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img, None)
    print(kp[0].pt)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    # draw only keypoints location, not size and orientation
    img2 = cv2.drawKeypoints(img, kp, outImage=None, color=(0,255,0), flags=0)
    plt.figure()
    plt.imshow(img2)
    plt.show()  

def test_feature_matcher():
    img1 = cv2.imread('frame1.jpg') 
    img2 = cv2.imread('frame5.jpg') 

    # Initiate SIFT detector
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    print(matches[1].trainIdx, matches[1].queryIdx, matches[1].imgIdx)
    print(kp1[matches[0].trainIdx].pt)
    
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1, img2, kp2, matches[:1], outImg=None, flags=2)

    plt.imshow(img3),plt.show()

def test_object_contains():
    


def main():
    
    #test_object_matcher()  
    #test_feature_detector()  
    test_feature_matcher()



    


if __name__ == '__main__':
    main()

    
        

        



