import typing

import cv2
import numpy as np
from numpy import random

mean_filter = np.ones((3, 3))

NO_BORDER = 0
ZERO_BORDER = 1
REPLICATE_BORDER = 2
REFLECT_BORDER = 3
PERIOD_BORDER = 4
REFLECT_BORDER_101 = 5

laplacian: np.ndarray = np.array([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]])

base_identity: np.ndarray = np.ones(shape=(5, 5), dtype=np.int8)

gauss_55 = np.array([[2., 7., 12., 7., 2.],
                       [7., 31., 52., 31., 7.],
                       [15., 52., 127., 52., 15.],
                       [7., 31., 52., 31., 7.],
                       [2., 7., 12., 7., 2.]], dtype=np.float64) / 423


gauss_33 = np.array([[1., 2., 1.],
                       [2., 4., 2.],
                       [1., 2., 1.]], dtype=np.float64) / 16


def Derivative(image: np.ndarray, dim="x", border=REPLICATE_BORDER) -> np.ndarray:
    if dim == "x":
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        return Filter(image, sobel_x, border)

    elif dim == "y":
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
        return Filter(image, sobel_y, border)
    elif dim == "":
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
        buff = Filter(image, sobel_x, border)
        return Filter(buff, sobel_y, border)


def MakeBorder(image, size=(1, 1), border=REPLICATE_BORDER) -> np.ndarray:
    h, w = image.shape[:2]
    y_center, x_center = size[0], size[1]
    if len(image.shape) == 3:
        padded = np.zeros(shape=(h + y_center * 2, w + x_center * 2, 3), dtype=np.uint8)
    else:
        padded = np.zeros(shape=(h + y_center * 2, w + x_center * 2), dtype=np.uint8)
    padded[size[0]: size[0] + h, size[1]: size[1] + w] = image
    if border == ZERO_BORDER:
        pass

    elif border == REPLICATE_BORDER:
        # UPPER BORDER
        for i in range(y_center):
            for j in range(x_center, w + x_center):
                padded[i][j] = padded[y_center][j]
        # LOWER BORDER
        for i in range(h + 1, h + y_center + 1):
            for j in range(x_center, w + x_center):
                padded[i][j] = padded[h][j]
        # LEFT BORDER
        for i in range(y_center, h + 1):
            for j in range(x_center):
                padded[i][j] = padded[i][x_center]
        # RIGHT BORDER
        for i in range(y_center, h + 1):
            for j in range(w + 1, w + x_center + 1):
                padded[i][j] = padded[i][w]

    elif border == REFLECT_BORDER:
        # UPPER BORDER
        for i in range(y_center):
            for j in range(x_center, w + x_center):
                padded[i][j] = padded[y_center + i][j]
        # LOWER BORDER
        for i in range(h + 1, h + y_center + 1):
            for j in range(x_center, w + x_center):
                padded[i][j] = padded[h - i][j]
        # LEFT BORDER
        for i in range(y_center, h + 1):
            for j in range(x_center):
                padded[i][j] = padded[i][x_center + j]
        # RIGHT BORDER
        for i in range(y_center, h + 1):
            for j in range(w + 1, w + x_center + 1):
                padded[i][j] = padded[i][w - j]

    elif border == PERIOD_BORDER:
        # UPPER BORDER
        for i in range(y_center):
            for j in range(x_center, w + x_center):
                padded[i][j] = padded[h - i][j]
        # LOWER BORDER
        for i in range(h + 1, h + y_center + 1):
            for j in range(x_center, w + x_center):
                padded[i][j] = padded[y_center + i - h - 1][j]
        # LEFT BORDER
        for i in range(y_center, h + 1):
            for j in range(x_center):
                padded[i][j] = padded[i][w - j]
        # RIGHT BORDER
        for i in range(y_center, h + 1):
            for j in range(w + 1, w + x_center + 1):
                padded[i][j] = padded[i][x_center + j - w - 1]

    elif border == REFLECT_BORDER_101:
        # UPPER BORDER
        for i in range(y_center):
            for j in range(x_center, w + x_center):
                padded[i][j] = padded[y_center + i + 1][j]
        # LOWER BORDER
        for i in range(h + 1, h + y_center + 1):
            for j in range(x_center, w + x_center):
                padded[i][j] = padded[h - i - 1][j]
        # LEFT BORDER
        for i in range(y_center, h + 1):
            for j in range(x_center):
                padded[i][j] = padded[i][x_center + j + 1]
        # RIGHT BORDER
        for i in range(y_center, h + 1):
            for j in range(w + 1, w + x_center + 1):
                padded[i][j] = padded[i][w - j - 1]
    return padded


def Filter(image: np.ndarray, kernel: np.ndarray, border: int = ZERO_BORDER) -> np.ndarray:
    blank_image = np.zeros_like(image, dtype=np.uint8)
    m, n = kernel.shape
    h, w = image.shape[:2]
    y_center, x_center = m // 2, n // 2
    padded = MakeBorder(image, (y_center, x_center), border=border)
    for y in range(h):
        for x in range(w):
            region = padded[y: y + m, x: x + n]
            if len(image.shape) == 3:
                r, g, b = 0, 0, 0
                for i in range(m):
                    for j in range(n):
                        r += region[i][j][0] * kernel[i][j]
                        g += region[i][j][1] * kernel[i][j]
                        b += region[i][j][2] * kernel[i][j]

                if r < 0:
                    r = 0
                if r > 255:
                    r = 255

                if g < 0:
                    g = 0
                if g > 255:
                    g = 255

                if b < 0:
                    b = 0
                if b > 255:
                    b = 255
                blank_image[y][x] = [r, g, b]
            else:
                s = 0
                for i in range(m):
                    for j in range(n):
                        s += region[i][j] * kernel[i][j]
                if s < 0:
                    s = 0
                if s > 255:
                    s = 255
                blank_image[y][x] = s

    return blank_image


def normalize(array: np.ndarray, lower, higher, ret_max=False) -> np.ndarray or (np.ndarray, float or int):
    maximum = -1
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i][j] > maximum:
                maximum = array[i][j]
            if array[i][j] < lower:
                array[i][j] = lower

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i][j] /= maximum
            array[i][j] *= higher

    if ret_max:
        maximum = lower - 1
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i][j] > maximum:
                    maximum = array[i][j]
        return array, maximum
    return array





def HarrisEdgeDetection(image: np.ndarray, kernel=gauss_55, k=0.04, threshold=0.5) -> np.ndarray:
    if len(image.shape) == 3:
        raise Exception(f"Incorrect format shape {image.shape}, convert to GRAY")
    h, w = image.shape
    center = kernel.shape[0] // 2
    der_x = Derivative(image, "x").astype(dtype=np.uint32)
    der_y = Derivative(image, "y").astype(dtype=np.uint32)
    Ixx = np.zeros_like(image, dtype=np.uint16)
    Iyy = np.zeros_like(image, dtype=np.uint16)
    Ixy = np.zeros_like(image, dtype=np.uint16)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            Ixx[i][j] = der_x[i][j] * der_x[i][j]
            Iyy[i][j] = der_y[i][j] * der_y[i][j]
            Ixy[i][j] = der_x[i][j] * der_y[i][j]

    responce = np.zeros_like(image, dtype=np.float64)
    for y in range(center, h - center):
        for x in range(center, w - center):
            Sxx, Syy, Sxy = 0, 0, 0
            windowIxx = Ixx[y - center: y + center + 1, x - center: x + center + 1]
            windowIyy = Iyy[y - center: y + center + 1, x - center: x + center + 1]
            windowIxy = Ixy[y - center: y + center + 1, x - center: x + center + 1]

            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]):
                    Sxx += kernel[i][j] * windowIxx[i][j]
                    Syy += kernel[i][j] * windowIyy[i][j]
                    Sxy += kernel[i][j] * windowIxy[i][j]

            det_M = Sxx * Syy - Sxy * Sxy
            trace_M = Sxx + Syy
            R = det_M - k * (trace_M ** 2)
            if R < 0:
                responce[y, x] = 0
            else:
                responce[y, x] = R

    # Normalize R
    responce, maximum = normalize(responce, 0, 255, ret_max=True)

    corners = np.zeros_like(responce, dtype=np.uint8)
    for i in range(corners.shape[0]):
        for j in range(corners.shape[1]):
            if responce[i][j] > threshold * maximum:
                corners[i][j] = int(responce[i][j])

    return corners


def NonMaxSupression(harris_responce: np.ndarray, ksize: int = 5) -> np.ndarray:
    h, w = harris_responce.shape
    center = ksize // 2
    non_max_sup = np.zeros_like(harris_responce, dtype=np.uint8)
    for y in range(center, h - center):
        for x in range(center, w - center):
            region = harris_responce[y - center: y + center + 1, x - center: x + center + 1]
            maximum = -1
            for i in range(ksize):
                for j in range(ksize):
                    if region[i][j] > maximum:
                        maximum = region[i][j]
            if harris_responce[y][x] == maximum:
                non_max_sup[y][x] = harris_responce[y][x]

    return non_max_sup


if __name__ == "__main__":
    img = cv2.imread("Rainier2.png", cv2.IMREAD_GRAYSCALE)
    som = 5
    val = 0.01
    tresh = 0.1
    imga = NonMaxSupression(HarrisEdgeDetection(img, k=val, threshold=tresh), som)
    cv2.imshow("dasd", imga)
    cv = cv2.cornerHarris(img, som, som, val)
    dst = cv2.dilate(cv, None)
    cv[dst > tresh * dst.max()] = 255
    cv2.imshow("cv", cv)

    for row in imga:
        for col in row:
            if col > 0:
                print(col)

    cv2.waitKey(0)
