import cv2
import numpy as np
from SMT import HarrisEdgeDetection, NonMaxSupression
from numpy.random import randint
import copy
import numpy.linalg as lg


def get_pos_desc(im1, im2, features_1, features_2, width=5):
    positions_1 = []
    for y in range(features_1.shape[0]):
        for x in range(features_2.shape[1]):
            if features_1[y][x] != 0 and 0 <= y - width and y + width < features_1.shape[0] and 0 <= x - width and x + width < features_1.shape[1]:
                mean = 0.
                std = 0.
                region = im1[y - width: y + width + 1, x - width: x + width + 1]
                region = region.astype(dtype=np.float64)
                for i in range(2*width + 1):
                    for j in range(2*width + 1):
                        mean += region[i][j] / ((2*width + 1) * (2*width + 1))
                for i in range(2*width + 1):
                    for j in range(2*width + 1):
                        std += (region[i][j] - mean)**2
                std = np.sqrt(std / ((2*width + 1) * (2*width + 1)))
                for i in range(2*width + 1):
                    for j in range(2*width + 1):
                        region[i][j] = (region[i][j] - mean) / std

                positions_1.append((x, y, region))

    positions_2 = []
    for y in range(features_2.shape[0]):
        for x in range(features_2.shape[1]):
            if features_2[y][x] != 0 and 0 <= y - width and y + width < features_2.shape[0] and 0 <= x - width and x + width < features_2.shape[1]:
                mean = 0.
                std = 0.
                region: np.ndarray = im2[y - width: y + width + 1, x - width: x + width + 1]
                region = region.astype(dtype=np.float64)
                for i in range(2*width + 1):
                    for j in range(2*width + 1):
                        mean += region[i][j] / ((2*width + 1) * (2*width + 1))
                for i in range(2*width + 1):
                    for j in range(2*width + 1):
                        std += (region[i][j] - mean) ** 2
                std = np.sqrt(std / ((2*width + 1) * (2*width + 1)))
                for i in range(2*width + 1):
                    for j in range(2*width + 1):
                        region[i][j] = (region[i][j] - mean) / std
                positions_2.append((x, y, region))

    return positions_1, positions_2


def find_matched_descriptors(feat_1, feat_2, width=5):
    matching = []
    if len(feat_1) > len(feat_2):
        feat_1_ = feat_2
        feat_2_ = feat_1
    else:
        feat_1_ = feat_1
        feat_2_ = feat_2

    for f_1 in feat_1_:
        best_sum = float("inf")
        best_point = None
        for f_2 in feat_2_:
            control_sum = 0.
            reg1 = f_1[2]
            reg_2 = f_2[2]
            for i in range(2*width + 1):
                for j in range(2*width + 1):
                    control_sum += (reg1[i][j] - reg_2[i][j])**2

            if control_sum < best_sum:
                best_sum = control_sum
                best_point = f_2
        if best_sum < 10:
            matching.append((f_1, best_point))

    return matching


def RANSACK(image1: np.ndarray, image2: np.ndarray, count_iterations: int, coridor: int, width=3) -> np.ndarray:
    base_features = NonMaxSupression(HarrisEdgeDetection(image1, threshold=0.2, k=0.04))
    dother_features = NonMaxSupression(HarrisEdgeDetection(image2, threshold=0.2, k=0.04))

    features_pos_1, features_pos_2 = get_pos_desc(image1, image2, base_features, dother_features, width=width)

    find = find_matched_descriptors(features_pos_1, features_pos_2, width=width)

    i = 0
    best_matrix = None
    max_popalo = 0
    while i < count_iterations:
        random_points = np.random.randint(0, len(find), size=3)
        while random_points[0] == random_points[1] or random_points[0] == random_points[2] or random_points[1] == random_points[2]:
            random_points = np.random.randint(0, len(find), size=3)

        M = np.array([[find[int(random_points[0])][0][0], find[int(random_points[0])][0][1], 1, 0, 0, 0],
                      [0, 0, 0, find[int(random_points[0])][0][0], find[int(random_points[0])][0][1], 1],
                      [find[int(random_points[1])][0][0], find[int(random_points[1])][0][1], 1, 0, 0, 0],
                      [0, 0, 0, find[int(random_points[1])][0][0], find[int(random_points[1])][0][1], 1],
                      [find[int(random_points[2])][0][0], find[int(random_points[2])][0][1], 1, 0, 0, 0],
                      [0, 0, 0, find[int(random_points[2])][0][0], find[int(random_points[2])][0][1], 1]
                      ], dtype=np.float64)

        b = np.array([find[int(random_points[0])][1][0],
                      find[int(random_points[0])][1][1],
                      find[int(random_points[1])][1][0],
                      find[int(random_points[1])][1][1],
                      find[int(random_points[2])][1][0],
                      find[int(random_points[2])][1][1]], dtype=np.float64)
        if lg.det(M) != 0:
            i += 1
            # A = np.matmul(np.matmul(lg.inv(np.matmul(lg.matrix_transpose(M), M)), lg.matrix_transpose(M)), b)
            A = np.matmul(lg.inv(M), b)
        else:
            continue
        # Матрицей я переношу с 1-ой фото на вторую
        popalo = 0
        for f in find:
            n_x = A[0] * f[0][0] + A[1] * f[0][1] + A[2]
            n_y = A[3] * f[0][0] + A[4] * f[0][1] + A[5]
            if abs(n_x - f[1][0]) < coridor and abs(n_y - f[1][1]) < coridor:
                popalo += 1

        if popalo > max_popalo:
            max_popalo = popalo
            best_matrix = A

    new_positions = []
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            n_x = best_matrix[0] * j + best_matrix[1] * i + best_matrix[2]
            n_y = best_matrix[3] * j + best_matrix[4] * i + best_matrix[5]
            new_positions.append((int(n_x), int(n_y)))

    print(new_positions)
    print(image1.shape, image2.shape)

    # matrix = np.array([[best_matrix[0], best_matrix[1], best_matrix[2]],
    #                    [best_matrix[3], best_matrix[4], best_matrix[5]]])
    # transformed_image2 = cv2.warpAffine(image2, matrix, (image1.shape[1], image1.shape[0]))
    # result = cv2.addWeighted(image1, 0.5, transformed_image2, 0.5, 0)
    # cv2.imshow('Result', result)



image_ = cv2.imread("Rainier1.png", cv2.IMREAD_GRAYSCALE)

image2_ = cv2.imread("Rainier2.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("1", image_)
cv2.imshow("2", image2_)

RANSACK(image_, image2_, 10000, 4)

cv2.waitKey(0)
