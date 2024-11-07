import cv2
import numpy as np


def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)


def add_non_overlapping_rectangle(rectangles, new_rect):
    for rect in rectangles:
        if is_overlapping(rect, new_rect):
            if rect[2] - rect[0] > new_rect[2] - new_rect[0]:
                return True
            else:
                rectangles.remove(rect)
                rectangles.append(new_rect)
                return False
    rectangles.append(new_rect)
    return True


image = cv2.imread('chess_8.jpg', cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(image, (5, 5), 0)

edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

squares_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

non_overlapping_squares = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    side = max(w, h)

    cx, cy = x + w // 2, y + h // 2

    top_left_x = cx - side // 2
    top_left_y = cy - side // 2

    new_square = (top_left_x, top_left_y, top_left_x + side, top_left_y + side)

    add_non_overlapping_rectangle(non_overlapping_squares, new_square)

non_overlapping_squares = sorted(non_overlapping_squares, key=lambda square: (square[0], square[1]))
summa = 0
between = 0
leng = len(non_overlapping_squares)
for i, rect in enumerate(non_overlapping_squares):
    summa += rect[2] - rect[0]
    if i + 1 < leng:
        between += non_overlapping_squares[i + 1][2] - rect[0]


for rect in non_overlapping_squares:
    cv2.rectangle(squares_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
    cv2.imshow('faas', squares_image)
summa = summa // len(non_overlapping_squares)
between = between // len(non_overlapping_squares)

# FIND FIRST ANGLE
def findFirst(image, side=2, tres= 100):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            check = np.ones((side * 2 + 1, side * 2 + 1), dtype=np.uint8)
            if image[i][j] >tres:
                check[side][side] = 0
                for k in range(-side, side + 1):
                    if 0 <= i + k < image.shape[0]:
                        for l in range(-side, side + 1):
                            if 0 <= j + l < image.shape[1]:
                                if k == 0 and l < 0:
                                    if image[i + k][j + l] < tres:
                                        check[side][side + l] = 0
                                elif k == 0 and l > 0:
                                    if image[i + k][j + l] > tres:
                                        check[side][side + l] = 0
                                elif l == 0 and k < 0:
                                    if image[i + k][j + l] < tres:
                                        check[side + k][side] = 0
                                elif l == 0 and k > 0:
                                    if image[i + k][j + l] > tres:
                                        check[side + k][side] = 0
            summa = 0
            for row in check:
                for col in row:
                    summa += col
            if summa == side * side * 4:
                return i, j

first = findFirst(image, 6, 200)

for i in range(0, image.shape[0], between + summa):
    for j in range(first[1] % (between + summa), image.shape[1], between + summa):
        if j < first[1]:
            cv2.rectangle(image, (i, j), (i + summa, j + summa), (0, 0, 255), 2)


cv2.imshow('Non-overlapping Squares', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
