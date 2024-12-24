#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob

# Определение размеров шахматной доски
CHECKERBOARD = (7, 7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = []

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None


images = glob.glob('C:/Users/User/Documents/PyProj/AgainMach/Photos/New/Cal_Photo/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)


cv2.destroyAllWindows()

h, w = img.shape[:2]

"""
Выполнение калибровки камеры с помощью
Передача значения известных трехмерных точек (объектов)
и соответствующие пиксельные координаты
обнаруженные углы (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

camera_matrix = mtx
print(mtx)
print(dist)

camera_distortion = dist

camera_rvecs = (np.array([[0.35489213],
                          [-0.22140186],
                          [0.0085083]]), np.array([[0.19452628],
                                                   [-0.28029741],
                                                   [0.00437356]]), np.array([[0.23228733],
                                                                             [0.30498164],
                                                                             [0.05765721]]), np.array([[-0.50079464],
                                                                                                       [-0.79114125],
                                                                                                       [1.40283552]]),
                np.array([[0.09978162],
                          [-0.31441551],
                          [1.56536252]]), np.array([[0.07105937],
                                                    [-0.38124153],
                                                    [1.56099916]]), np.array([[0.17826471],
                                                                              [-0.23240602],
                                                                              [1.53891274]]), np.array([[0.24700102],
                                                                                                        [-0.27459891],
                                                                                                        [1.51650143]]),
                np.array([[0.50239007],
                          [-0.09147182],
                          [0.07087062]]), np.array([[0.01041597],
                                                    [-0.46913666],
                                                    [0.00963228]]), np.array([[0.01004359],
                                                                              [-0.46865357],
                                                                              [0.00980562]]), np.array([[0.81430126],
                                                                                                        [-0.06426537],
                                                                                                        [0.03261915]]))
