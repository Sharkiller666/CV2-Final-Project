import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from numpy.linalg import inv
from numpy import linalg

def computeR(A, B):
    nA = A / sum(A)
    nB = B / sum(B)

    v = np.cross(nA, nB)
    s = linalg.norm(v)
    c = np.dot(nA, nB)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    return np.eye(3) + vx + (vx @ vx) * ((1 - c) / s*s )


def Get_R(A,B):

    #get unit vectors
    uA = A/np.sqrt(np.sum(np.square(A)))
    uB = B/np.sqrt(np.sum(np.square(B)))

    #get products
    cos_t = np.sum(uA * uB)
    sin_t = np.sqrt(np.sum(np.square(np.cross(uA,uB)))) #magnitude

    #get new unit vectors
    u = uA
    v = uB - np.sum(uA * uB)*uA
    v = v/np.sqrt(np.sum(np.square(v)))
    w = np.cross(uA, uB)
    w = w/np.sqrt(np.sum(np.square(w)))

    #get change of basis matrix
    C = np.array([u, v, w])

    #get rotation matrix in new basis
    R_uvw = np.array([[cos_t, -sin_t, 0],
                      [sin_t, cos_t, 0],
                      [0, 0, 1]])

    #full rotation matrix
    R = C.T @ R_uvw @ C
    #print(R)
    return R

if __name__ == '__main__':
    alpha = np.pi / 6

    # A = np.array([0, 1, 0])


    A = np.array([0, 1, 0])

    rot_alpha = np.array([[math.cos(alpha), -math.sin(alpha), 0],
                          [math.sin(alpha), math.cos(alpha), 0],
                          [0, 0, 1]])

    C = rot_alpha @ A
    print(C)

    # Wall
    # [-0.079   0.1204  0.9896]
    T = np.array([-0.079, 0.9896, 0.1204])
    R = Get_R(C, T)

    print(R)

    calib = R @ C

    print(calib)

    for e in calib:
        print(float(e))


    # A = np.array([-0.5, 0.8660254, 0])
    # B = np.array([0, 1, 0])

    # A:   [-0.5 0.866, 0]
    # B:   [0, 1, 0]
    # R:   [ 3.54903818  1.36602542  0.]
    #      [-1.36602542  3.54903818  0.]
    #      [ 0.          0.          1.]

    # print(computeR(A, B))
    #
    #
    # print(computeR(A, B) @ A)