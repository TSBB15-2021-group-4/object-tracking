#!/usr/bin/env python3

import numpy as np

def get_ground_model(X, Y, U, V):
    """
    Takes 4 world coordinates X, Y and corresponding pixel coordinates U,V and gives out
    a homography for transforming pixel coordinates to world coordinates.
    Author: Hoang Tran 2021-02-12
    """

    assert len(X) == 4
    assert len(Y) == 4
    assert len(U) == 4
    assert len(V) == 4

    D = np.array([[X[0], Y[0], 1, 0, 0, 0, -U[0] * X[0], -U[0] * Y[0]],
                  [0, 0, 0, X[0], Y[0], 1, -V[0] * X[0], -V[0] * Y[0]],
                  [X[1], Y[1], 1, 0, 0, 0, -U[1] * X[1], -U[1] * Y[1]],
                  [0, 0, 0, X[1], Y[1], 1, -V[1] * X[1], -V[1] * Y[1]],
                  [X[2], Y[2], 1, 0, 0, 0, -U[2] * X[2], -U[2] * Y[2]],
                  [0, 0, 0, X[2], Y[2], 1, -V[2] * X[2], -V[2] * Y[2]],
                  [X[3], Y[3], 1, 0, 0, 0, -U[3] * X[3], -U[3] * Y[3]],
                  [0, 0, 0, X[3], Y[3], 1, -V[3] * X[3], -V[3] * Y[3]],
                  ])

    f = np.array([U[0], V[0], U[1], V[1], U[2], V[2], U[3], V[3]])
    f = f.T

    c = np.linalg.pinv(D) @ f
    c = np.append(c, 1)
    C = np.reshape(c, (3, 3))
    pixel2world_homograpy = np.linalg.inv(C)

    return pixel2world_homograpy
