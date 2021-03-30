#!/usr/bin/env python3

import numpy as np
import cv2
from matplotlib import pyplot as plt
from Object import Object



def test_object_contains():
    obj = Object(box_width=5, box_height=10, pos_x=1, pos_y=1, center_x=0, center_y=0, silhouette=0)
    
    # 1 <= point.x <= 6  ,   1 <= point.y <= 11
    
    assert(obj.contains([1,1]))

    assert(obj.contains([11,6]))

    assert(obj.contains([8,6]))

    assert(obj.contains([63,114]) == False)

def main():
      
    test_object_contains()



    


if __name__ == '__main__':
    main()
