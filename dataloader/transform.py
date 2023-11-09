import numpy as np


def trans_rotate(pts, rotation):
    T = [[np.cos(rotation), -np.sin(rotation), 0],
        [np.sin(rotation), np.cos(rotation), 0],
        [0, 0, 1]]
    pts_1 = np.dot(T, np.concatenate((pts.T, np.ones((1, pts.shape[0]))), axis=0))[:2, :].T
    return pts_1


def trans_translate(pts, local_pt):
    T = [[1, 0, -local_pt[0]],
         [0, 1, -local_pt[1]],
         [0, 0, 1]]
    pts_1 = np.dot(T, np.concatenate((pts.T, np.ones((1, pts.shape[0]))), axis=0))[:2, :].T
    return pts_1


def world_to_frenet(pts, local_pt, rotation):
    pts_1 = trans_translate(pts, local_pt)
    pts_frenet = trans_rotate(pts_1, rotation)
    return pts_frenet