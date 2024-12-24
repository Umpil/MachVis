# import cv2
# import numpy as np
# from CalibreCamera import camera_matrix, camera_distortion

# image = cv2.imread('C:/Users/User/Documents/PyProj/AgainMach/Photos/New/Good/FromChess.png')

# approx = np.array([[32,22],[1196,82], [1190,1253], [19,1253]])
# h,  w = image.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, camera_distortion, (w,h), 1, (w,h))

# image = cv2.undistort(image, camera_matrix, camera_distortion, None, newcameramtx)
# cv2.imwrite("C:/Users/User/Documents/PyProj/AgainMach/Photos/New/Good/UdistChess.png", image)
# src_points = np.float32(approx.reshape(4, 2))


# width = image.shape[0] 
# height = image.shape[1] 
# dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])


# matrix = cv2.getPerspectiveTransform(src_points, dst_points)
# print(matrix)


# warped_image = cv2.warpPerspective(image, matrix, (width, height))


# cv2.imwrite('warped_chessboard.jpg', warped_image)
# cv2.imshow('Warped Chessboard', warped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import cv2
import numpy as np
gray = cv2.imread('C:/Users/User/Documents/PyProj/AgainMach/Photos/New/Lited/LitedInvHor.jpg', cv2.IMREAD_GRAYSCALE)[:1700, 1700:]
image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# Найдите контуры
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Для хранения результатов
curvatures = []

# Обработайте каждый контур
for contour in contours:
    # Получите координаты точек контура
    points = contour[:, 0, :]
    
    # Вычислите параметры кривизны
    # Например, можно использовать метод наименьших квадратов для подгонки кривой
    if len(points) >= 3:  # Убедитесь, что достаточно точек
        # Полином 2-й степени (парабола) для аппроксимации
        fit = np.polyfit(points[:, 0], points[:, 1], 2)
        
        # Вычисляем кривизну (вторая производная)
        curvature = 2 * fit[0]  # Кривизна для параболы
        curvatures.append(curvature)

# Выводим результаты
for i, curvature in enumerate(curvatures):
    print(f"Кривизна контура {i}: {curvature}")

# Визуализация контуров и кривизны
for contour in contours:
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

cv2.imshow('Contours with Curvature', image)
cv2.waitKey(0)
cv2.destroyAllWindows()