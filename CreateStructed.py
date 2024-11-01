import cv2
import numpy as np


def CreateStructedLight(size, step, dim=0, inverse=False):
    blank_image = np.zeros(shape=(size[0], size[1], 3), dtype=np.uint8)
    if inverse:
        sets = 0
    else:
        sets = 1
    if dim == 0:
        lasts = size[0] % step
        for i in range(0, size[0] - lasts, step):
            if (i // step) % 2 == sets:
                for j in range(size[1]):
                    for k in range(step):
                        blank_image[i + k][j] = np.array([255, 255, 255])
    elif dim == 1:
        lasts = size[1] % step
        print(lasts)
        for i in range(size[0]):
            for j in range(0, size[1] - lasts, step):
                if (j // step) % 2 == sets:
                    for k in range(step):
                        blank_image[i][j + k] = np.array([255, 255, 255], dtype=np.uint8)
    elif dim == -1:
        lasts_row = size[0] % step
        lasts_col = size[1] % step
        for i in range(0, size[0] - lasts_row, step):
            if (i // step) % 2 == sets:
                for j in range(0, size[1] - lasts_col, step):
                    if (j // step) % 2 == sets:
                        for k in range(step):
                            for l in range(step):
                                blank_image[i + k][j + l] = np.array([255, 255, 255], dtype=np.uint8)
    else:
        return None

    return blank_image


shape = (1080, 1920)
step = 4
image = CreateStructedLight(shape, step, 0)
cv2.imshow("horiz", image)
cv2.imwrite(f"Horizontal_{step}.png", image)
image = CreateStructedLight(shape, step, 1)
cv2.imshow("vert", image)
cv2.imwrite(f"Vertical_{step}.png", image)
image = CreateStructedLight(shape, step, -1)
cv2.imshow("chess", image)
cv2.imwrite(f"Chess_{step}.png", image)

cv2.waitKey(0)
cv2.destroyAllWindows()