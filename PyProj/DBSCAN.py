import cv2
import numpy as np
from numpy import random
from itertools import cycle
import time

def DBSCAN(Points, eps, m, distance):
    Noise = 0
    C = 0

    VisitedPoints = set()
    ClusteredPoints = set()
    Clusters = {Noise: []}


    def region_query(point):
        return [q for q in Points if distance(point, q, eps)]

    def expand_cluster(point, neighbours):
        if C not in Clusters:
            Clusters[C] = []
        
        Clusters[C].append(point)
        ClusteredPoints.add(point)
        while neighbours:
            q = neighbours.pop()
            if q not in VisitedPoints:
                VisitedPoints.add(q)
                neighbours_q = region_query(q)
                if len(neighbours_q) > m:
                    neighbours.extend(neighbours_q)
            if q not in ClusteredPoints:
                ClusteredPoints.add(q)
                Clusters[C].append(q)
                if q in Clusters[Noise]:
                    Clusters[Noise].remove(q)
    
    print(len(Points))
    for i, Point in enumerate(Points):
        if Point in VisitedPoints:
            continue
        print(i)
        VisitedPoints.add(Point)
        neighbours = region_query(Point)
        if len(neighbours) < m:
            Clusters[Noise].append(Point)
        else:
            C += 1
            expand_cluster(Point, neighbours)

    return Clusters

def norm(x1, x2, eps):
    return ((np.abs((x1[0] - x2[0])) + np.abs(x1[1] -x2[1])) < eps) and (x1[2] == x2[2])

if __name__ == "__main__":
    colors_bgr = [
    (255, 0, 0),    # Красный
    (0, 255, 0),    # Зеленый
    (0, 0, 255),    # Синий
    (255, 255, 0),  # Желтый
    (0, 255, 255),  # Циан
    (255, 0, 255),  # Магента
    (128, 128, 128), # Серый
    (255, 165, 0),   # Оранжевый
    (75, 0, 130),    # Индиго
    (238, 130, 238), # Фиолетовый
    (255, 192, 203), # Розовый
    (0, 128, 128),   # Бирюзовый
    (210, 180, 140), # Тан
    (160, 32, 240),  # Фиолетовый
    (34, 139, 34),   # Лесной зеленый
    (255, 20, 147),  # Deep Pink
    (218, 112, 214), # Orchid
    (255, 228, 196), # Bisque
    (255, 99, 71),   # Tomato
    (135, 206, 250), # Light Sky Blue
    (240, 230, 140), # Khaki
    (144, 238, 144), # Light Green
    (255, 228, 181), # Papaya Whip
    (100, 149, 237), # Cornflower Blue
    (255, 218, 185), # Peach Puff
    (186, 85, 211),  # Medium Orchid
    (221, 160, 221), # Plum
    (176, 224, 230), # Powder Blue
    (255, 127, 80),  # Coral
    (0, 206, 209),   # Dark Turquoise
    (186, 222, 222), # Light Steel Blue
    (205, 92, 92),   # Indian Red
    (205, 133, 63),  # Peru
    (255, 245, 238), # Seashell
    (72, 61, 139),   # Dark Slate Blue
    (139, 69, 19),   # Saddle Brown
    (70, 130, 180),  # Steel Blue
    (210, 105, 30),  # Chocolate
    (255, 105, 180), # Hot Pink
    (135, 206, 235), # Sky Blue
    (255, 239, 0),   # Bright Yellow
    (0, 191, 255),   # Deep Sky Blue
    (50, 205, 50),   # Lime Green
    (255, 160, 122), # Light Salmon
    (173, 216, 230), # Light Blue
    (240, 128, 128), # Light Coral
    (135, 204, 250), # Light Cyan
    (152, 251, 152), # Pale Green
    (255, 228, 225), # Misty Rose
    (123, 104, 238), # Medium Slate Blue
    (186, 85, 211),   # Medium Violet Red
    (240, 248, 255), # Alice Blue
    (255,215 ,0)     # Gold
]
    image = cv2.imread("AgainMach/Photos/New/Lited/LitedInvHor.jpg", cv2.IMREAD_GRAYSCALE)
    image = image[350:450, 2050:2250]
    flat_image = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            flat_image.append((i, j, int(image[i][j])))
    flat_image = flat_image
    time_start = time.time()
    clustered_data = DBSCAN(flat_image, 5, 8, norm)
    print(f"Time past: {int((time.time() - time_start) / 60)}")
    print(f"Num of cluster: {len(clustered_data)}")
    new_image = np.zeros(shape=(image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for c, points in zip(cycle(colors_bgr), clustered_data.values()):
        for p in points:
            new_image[p[0]][p[1]] = c
    cv2.imwrite("AgainMach/Photos/New/DBSCAN.png", new_image)
    cv2.imshow("colored_data", new_image)
    cv2.imshow("BASED", image)
    cv2.waitKey(0)
