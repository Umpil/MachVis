import cv2
import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import SMT as sm

CalibreDistance = np.float64(45.0)
ideal_lenght = 16
PerspectiveMatrix = np.array([[1.04417967e+00, 1.10270801e-02, -3.36563451e+01], 
                              [-5.15814334e-02, 1.00067981e+00, -2.03643499e+01], 
                              [-4.20386580e-05, 4.21966799e-06, 1.00000000e+00]], dtype=np.float64)

camera_matrix = np.array([[7.76291622e+03, 0.00000000e+00, 2.49924143e+03],
                          [0.00000000e+00, 7.69456362e+03, 1.83026350e+03],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

camera_distortion = np.array([[-0.0380533, 0.04055333, 0.01010542, 0.00684829, 2.00261]])

def get_max_min_mask(im_array):
    mask_maximum = np.zeros_like(im_array[0], dtype=np.uint8)
    mask_minimum = np.zeros_like(im_array[0], dtype=np.uint8)
    for row in range(im_array[0].shape[0]):
        for col in range(im_array[0].shape[1]):
            val_range = []
            for item in im_array:
                val_range.append(item[row][col])
            maxim = -1
            minim = 258
            for item in val_range:
                if item > maxim:
                    maxim = item
                if item < minim:
                    minim = item
            mask_maximum[row][col] = maxim
            mask_minimum[row][col] = minim

    return mask_maximum, mask_minimum


def find_Ld_Lg_pixels(im_array):
    ret_mask = np.zeros(shape=(im_array[0].shape[0], im_array[0].shape[1], 2), dtype=np.float64)
    mask_max, mask_min = get_max_min_mask(gray_l_d_v_vi_h_hi)
    mask_max = mask_max.astype(dtype=np.float64)
    mask_min = mask_min.astype(dtype=np.float64)

    for i in range(mask_max.shape[0]):
        for j in range(mask_max.shape[1]):
            if im_array[0][i][j] == 0:
                continue
            b = (np.float64(im_array[0][i][j]) - np.float64(im_array[1][i][j]))/np.float64(im_array[0][i][j])
            L_d = (np.float64(mask_max[i][j]) - np.float64(mask_min[i][j]))/(1.0-b)
            L_g = 2.0 * (np.float64(mask_min[i][j]) - b * np.float64(mask_max[i][j]))/(1.0-b)
            ret_mask[i][j] = np.array([L_d, L_g])
    
    return ret_mask


def find_Lited_pixels(image, inverse_image, Ld_Lg, eps=5, m=15, ret_gray=True):
    if ret_gray:
        ret_pixels = np.zeros_like(image)
    else:
        ret_pixels = np.zeros(shape=(image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    #Lit 255 or 0 0 255
    #Uncknown 125 or 255 0 0
    #Unlit 0 or 0 0 0

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if Ld_Lg[i][j][0] < m:
                if ret_gray:
                    ret_pixels[i][j] = 125
                else:
                    ret_pixels[i][j] = [255, 0, 0]
            elif Ld_Lg[i][j][0] > Ld_Lg[i][j][1] + eps and np.float32(image[i][j]) > np.float32(inverse_image[i][j]) + eps:
                if ret_gray:
                    ret_pixels[i][j] = 255
                else:
                    ret_pixels[i][j] = [0, 0, 255]
            elif Ld_Lg[i][j][0] > Ld_Lg[i][j][1] and np.float32(image[i][j]) > np.float32(inverse_image[i][j]) + eps:
                if ret_gray:
                    ret_pixels[i][j] = 255
                else:
                    ret_pixels[i][j] = [0, 0, 255]
            elif Ld_Lg[i][j][0] > Ld_Lg[i][j][1] + eps and np.float32(image[i][j]) + eps < np.float32(inverse_image[i][j]):
                if ret_gray:
                    ret_pixels[i][j] = 0
                else:
                    ret_pixels[i][j] = [0, 0, 0]
            elif np.float32(image[i][j]) + eps < Ld_Lg[i][j][0] and np.float32(inverse_image[i][j]) > Ld_Lg[i][j][1] + eps:
                if ret_gray:
                    ret_pixels[i][j] = 0
                else:
                    ret_pixels[i][j] = [0, 0, 0]
            elif np.float32(image[i][j]) > Ld_Lg[i][j][1] + eps and np.float32(inverse_image[i][j]) + eps > Ld_Lg[i][j][0]:
                if ret_gray:
                    ret_pixels[i][j] = 255
                else:
                    ret_pixels[i][j] = [0, 0, 255]
            else:
                if ret_gray:
                    ret_pixels[i][j] = 125
                else:
                    ret_pixels[i][j] = [255, 0, 0]
    return ret_pixels


def findPointers(src_im, dst_im, ksize=3):
    dst = cv2.cornerHarris(np.float32(dst_im), 2, 3, 0.04)
    src = cv2.cornerHarris(np.float32(src_im), 2, 3, 0.04)

    feat_1, feat_2 = [], []
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if dst[i][j] > 0.01 * dst.max():
                if i - ksize >= 0 and i + ksize < dst.shape[0] and j - ksize >= 0 and j + ksize < dst.shape[1]:
                    patch = dst[i - ksize: i + ksize + 1, j - ksize: j + ksize + 1].astype(dtype=np.float32)
                    feat_1.append((i, j, patch))

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if src[i][j] > 0.01 * src.max():
                if i - ksize >= 0 and i + ksize < src.shape[0] and j - ksize >= 0 and j + ksize < src.shape[1]:
                    patch = dst[i - ksize: i + ksize + 1, j - ksize: j + ksize + 1].astype(dtype=np.float32)
                    feat_2.append((i, j, patch))

    if len(feat_1) < len(feat_2):
        feat_1_ = feat_1
        feat_2_ = feat_2
    else:
        feat_1_ = feat_2
        feat_2_ = feat_1

    matching = []
    #SSIM
    c1 = np.float32(0.01*255*255*0.01)
    c2 = np.float32(0.03*255*255*0.03)
    for f_1 in feat_1_:
        best_sum = -float("inf")
        best_point = None
        for f_2 in feat_2_:
            control_sum = 0.
            reg1 = f_1[2]
            mean_1 = reg1.mean()
            std_1 = reg1.std()
            reg2 = f_2[2]
            mean_2 = reg2.mean()
            std_2 = reg2.std()
            cov = np.float32(0.0)
            for i in range(2*ksize + 1):
                for j in range(2*ksize + 1):
                    cov += (reg1[i][j] - mean_1) * (reg2[i][j] - mean_2)/((2*ksize + 1) * (2*ksize + 1))

            control_sum = (2.0*mean_1*mean_2 + c1)*(2.0*cov + c2)/((mean_1**2 + mean_2**2 + c1)*(std_1**2 + std_2**2 + c2))
        
            if control_sum > best_sum:
                best_sum = control_sum
                best_point = f_2

        if best_sum > 0.9:
            matching.append((f_1, best_point))
    return matching


def check_point(point, prevs,  i, j, distance=2):
    if i - 2 < 0 or j - 2 < 0:
        return True
    return (lg.norm(point - prevs[i-1][j]) > distance and lg.norm(point - prevs[i][j-1]) > distance and lg.norm(point - prevs[i-1][j-1]) > distance and lg.norm(point - prevs[i][j-2]) > distance and lg.norm(point - prevs[i-2][j]) > distance and lg.norm(point - prevs[i-2][j-1]) > distance and lg.norm(point - prevs[i-1][j-2]) > distance)


def get_point_cloud(straight, inverse, c_m, distance=2):

    return straight


def filter_cloud(cloud, width=11, count_sigma=3):
    x = cloud[:, 0]
    y = cloud[:, 1]
    max_w = int(np.max(x)) + 1
    max_h = int(np.max(y)) + 1
    array = np.zeros(shape=(max_h, max_w), dtype=np.float64)
    for point in cloud:
        if array[int(point[1])][int(point[0])] !=0:
            array[int(point[1])][int(point[0])] = (point[2] + array[int(point[1])][int(point[0])])/2 
        else:
            array[int(point[1])][int(point[0])] = point[2]
    
    mean_array = array.copy()
    for i in range(width//2, array.shape[0] - width//2):
        for j in range(array.shape[1]):
            reg = array[i-width//2: i+width//2 + 1, j - width//2:j+width//2 + 1]
            mean = reg.mean()
            std = reg.std()
            threshold = count_sigma*std
            anomalies = np.where((reg > mean + threshold) | (reg < mean - threshold))
            for anomaly in zip(*anomalies):
                mean_array[i + anomaly[0] - width//2][j + anomaly[1] - width // 2] = mean
                    
    ret_array = []

    # for i in range(mean_array.shape[0]):
    #     for j in range(mean_array.shape[1]):
    #         if not np.isclose(mean_array[i][j], 0):
    #             ret_array.append([j, i, np.uint16(mean_array[i][j])])
    
    for i in range(mean_array.shape[0]):
        for j in range(mean_array.shape[1]):
            window = mean_array[i-width//2: i+width//2 + 1, j - width//2:j+width//2 + 1].astype(dtype=np.float64)
            mean = window.mean()
            if np.isnan(mean):
                mean = 0
            ret_array.append([j, i, np.float64(mean)])

    return np.array(ret_array, dtype=np.float64)
        
    # full_cloud = np.zeros(shape=(len(horizontal) + len(vertical), 3))
    # full_cloud = np.concatenate((horizontal, vertical), axis=0)
    
    # full_cloud_sorted = full_cloud[np.lexsort((full_cloud[:, 1], full_cloud[:, 0]))]

    # unique_points = {}
    # for point in full_cloud_sorted:
    #     k = (point[0], point[1])
    #     if k in unique_points:
    #         if point[2] > unique_points[k][2]:
    #             unique_points[k] = point
    #     else:
    #         unique_points[k] = point

    # unique_array = np.array(list(unique_points.values()))

    # full_cloud_sorted = unique_array[np.lexsort((unique_array[:, 1], unique_array[:, 0]))]
    # filtered_points = [item for item in full_cloud_sorted[0:winsize//2]]
    # central_point = full_cloud_sorted[winsize//2]

    # ww = winsize//2 + 1
    # for i in range(ww, len(full_cloud_sorted) - ww - 1):
    #     bad = False
    #     for j in range(1, ww):
    #         if not lg.norm(full_cloud_sorted[i-j] - central_point) > distance or not lg.norm(full_cloud_sorted[i+j] - central_point) > distance:
    #             bad = True
    #     central_point = full_cloud_sorted[i]

    #     if not bad:
    #         filtered_points.append(full_cloud_sorted[i])

    # return full_cloud_sorted
    # Что-то около n!
    # ret_cloud = [horizontal[0]]
    # for point in horizontal:
    #     if all(lg.norm(point - p) >= distance for p in ret_cloud):#all(np.sqrt((point[0] - p[0])**2 + (point[1] - p[1])**2 + (point[2] - p[2])**2) >= distance for p in ret_cloud):
    #         ret_cloud.append(point)

    # print("Half")
    # for point in vertical:
    #     if all(lg.norm(point - p) >= distance for p in ret_cloud):
    #         ret_cloud.append(point)
                    
    # return np.array(ret_cloud, dtype=np.float64)

def get_good_lited(straight, inverse):
    st_k = 0
    st_inv = 0
    for i in range(straight.shape[0]):
        for j in range(straight.shape[1]):
            if straight[i][j] == 255:
                st_k +=1
            if inverse[i][j] == 255:
                st_inv += 1

    if st_k > st_inv:
        return straight.copy(), 0
    else:
        return inverse.copy(), 1
    
def find_pattern(straight, inverse, dim="y"):
    work_im, ret_im = get_good_lited(straight, inverse)
    if dim == "y":
        sobel = cv2.Sobel(work_im, cv2.CV_64F, 1, 0, ksize=5)[0:160, 1560:1750]
    elif dim == "x":
        sobel = cv2.Sobel(work_im, cv2.CV_64F, 0, 1, ksize=5)[0:160, 1560:1750]
    else:
        return
    sobel = np.clip(sobel, 0, 255)
    lines = cv2.HoughLinesP(sobel.astype(dtype=np.uint8), 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    angles = []
    max_angle = np.tan(np.pi * 22. /180.)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = (y2-y1)/(x2-x1)
        if max_angle > abs(angle) > 10**(-10):
            angles.append(angle)
    angles = np.array(angles, dtype=np.float64)
    if len(angles) == 0:
        angle = 0
    else:
        angle = angles.mean()
    print("angle: ", angle)
    sizes = (16, 18, 19, 20)
    shifts = (17, 19, 20, 21)
    best_pattern = None
    best_size = None
    best_sum = float("inf")
    best_shift = None
    work_zone = work_im[0:160, 1560:1750]
    for sn,size in enumerate(sizes):
        for shift in range(shifts[sn]):
            pattern_sample = np.zeros_like(work_im)

            if dim == "x":
                last = pattern_sample.shape[0] % size
                for y in range(-int(angle*pattern_sample.shape[0] + shift), pattern_sample.shape[0] - last, size):
                    if ((int(abs(y) + shift) // size) ) % 2 == 0:
                        for x in range(pattern_sample.shape[1]):
                                for k in range(size):
                                    if 0<=int(angle * x) + y+k < pattern_sample.shape[0]:
                                        pattern_sample[int(angle * x) + y+k][x] = 255

            if dim == "y":
                last = pattern_sample.shape[1] % size
                for y in range(-int(angle*pattern_sample.shape[0] - 1), pattern_sample.shape[0] - last, size):
                    for x in range(pattern_sample.shape[1]):
                            if ((x // size) + shift) % 2 == 0:
                                for k in range(size):
                                    if 0<=int(angle * x) + y+k < pattern_sample.shape[0]:
                                        pattern_sample[int(angle * x) + y+k][x] = 255
            
            control_im = 0.0
            work_sample = pattern_sample[0:160, 1560:1750]
            for i in range(2, work_sample.shape[0] - 2, 2):
                for j in range(2, work_sample.shape[1] - 2, 2):
                    reg1 = work_sample[i-2:i+2 + 1, j-2:j+2+1].astype(dtype=np.float64)
                    if np.abs(reg1.std()) < 0.001:
                        reg1 = reg1 - reg1.mean()
                    else:
                        reg1 = (reg1 - reg1.mean()) / reg1.std()
                    reg2 = work_zone[i-2:i+2 + 1, j-2:j+2+1].astype(dtype=np.float64)
                    if np.abs(reg2.std()) < 0.001:
                        reg2 = reg2 - reg2.mean()
                    else:
                        reg2 = (reg2 - reg2.mean()) / reg2.std()
                    control_im += np.sum((reg1 - reg2)**2)

            if control_im < best_sum:
                best_sum = control_im
                best_pattern = pattern_sample
                best_size = size
                best_shift = shift

    return best_size, best_pattern, best_shift, ret_im, angle


def find_curves(image, size, shift, angle, dim="x", inv=False, zazor=2):
    if dim == "x":
        print("Finding horizontal edges")
        if inv:
            sets = 1
        else:
            sets = 0
        cloud = []
        edges = sm.Canny(image, 251, 255)
        # edges = sm.DeleteNoNeigh(edges, times=2)
        #edges = sm.ConnectLeftRight(edges)
        edges = edges.astype(dtype=np.uint8)
        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                if edges[i][j] != 0 and edges[i][j] != 255:
                    edges[i][j] = 0
        
        patter = np.zeros_like(edges, dtype=np.uint16)
        last = patter.shape[0] % size
        for y in range(-int(angle*patter.shape[0] + shift), patter.shape[0] - last, size):
            if ((int(abs(y) + shift) // size) ) % 2 == sets:
                for x in range(patter.shape[1]):
                        for k in range(size):
                            if 0<=int(angle * x) + y+k < patter.shape[0]:
                                patter[int(angle * x) + y+k][x] = 255

        patter_edges = sm.Canny(patter, 251, 255)
        # if inv:
        #     cv2.imshow("image", image)
        #     cv2.imshow("ed", patter_edges)
        #     cv2.imshow("patt", patter.astype(dtype=np.uint8))
        #     cv2.waitKey(0)
        # cloud = []
        if image.shape[0] > image.shape[1]:
            const = image.shape[0] // 2
        else:
            const = image.shape[1] //2
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] != 0:
                    k = 0
                    while patter_edges[i + k][j] == 0:
                        k -= 1
                        if i + k < 0:
                            break
                    if i+k > 0:
                        if -k < 9 and image[i+k][j] == 255:
                            k -= 4
                            if i+k > 0:
                                while patter_edges[i + k][j] == 0:
                                    k -= 1
                                    if i + k < 0:
                                        break
                    if 0<=-k < 9 or -k > 40:
                        k= 0

                    #     if image[i + k - 1][j] != 0:
                    #         k -= 3
                    #         if i + k >=0:
                    #             while patter_edges[i + k][j] == 0:
                    #                 k-=1
                    #                 if i + k < 0:
                    #                     break
                    # if -k > size + zazor:
                    #     k = -(size + zazor)
                    # const*np.cos(np.pi*(-k/(size + 2))/2
                    cloud.append([j, i, -5*k])
                    
              
        return np.array(cloud, dtype=np.uint16)
        # center_x = patter_edges.shape[1]//2
        # betweens = []
        # for i in range(0, patter_edges.shape[0], st):
        #     if patter_edges[i][center_x] != 0:
        #         betweens.append((i-2, i+st, i//st))
        
        # for i in range(1, edges.shape[0]):
        #     for j in range(edges.shape[1]):
        #         if edges[i][j] != 0 and edges[i-1][j] == 0:
        #             num_line = None
        #             for bet in betweens:
        #                 if bet[0] <= j <= bet[1]:
        #                     num_line = bet[2]
        #             if num_line:
        #                 cloud.append([j, i, 50 * (j - bet[0])*np.cos((j-bet[0])/ (bet[1] - j))])

        # return np.array(cloud, dtype=np.float64)
        # cv2.imshow("edge", edges)
        # cv2.imshow("pat", patter_edges)
        # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # img = edges.copy()
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # print(len(contours))
        # for i in range(len(contours)):
        #     cont = contours[i]
        #     if len(cont) > 80:
        #         cv2.drawContours(img, [cont], 0, (0, 255, 0), 3)
        # cv2.imshow("dasd", img)
        # h = image.shape[0]
        # a_range, b_range, max_c = np.linspace(0, 0.025, 250), np.linspace(0, 0.03, 200), h
        # accumulator = np.zeros(shape=(len(a_range), len(b_range), max_c + 1), dtype=np.int32)
        # y_index, x_index = np.where(edges > 0)
        #
        # print("Voting")
        # for x, y in zip(x_index, y_index):
        #     for a_id, a in enumerate(a_range):
        #         for b_id, b in enumerate(b_range):
        #             c = y - (a * (x**2) + b*x)
        #             if 0 <= c < max_c:
        #                 accumulator[a_id][b_id][int(c)] +=1
        #
        # parabolas = np.argwhere(accumulator > treshold)
        # plt.imshow(image, cmap='gray')
        # for parabola in parabolas:
        #     a = a_range[parabola[0]]
        #     b = b_range[parabola[1]]
        #     c = parabola[2]
        #
        #     x_vals = np.linspace(0, image.shape[1], num=100)
        #     y_vals = a * (x_vals ** 2) + b * x_vals + c
        #
        #     plt.plot(x_vals, y_vals, color='red')
        # plt.show()
        # print("out")
        # fil = np.array([[1, 1, 0, 1, 1],
        #                [1, 1, 0, 1, 1],
        #                [0, 0, 0, 0, 0],
        #                [1, 1, 0, 1, 1],
        #                [1, 1, 0, 1, 1]], dtype=np.float64)
        # fil = np.array([[1, 0, 1],
        #                [0, 0, 0],
        #                [1, 0, 1]], dtype=np.float64)
        # fil = fil / 16
        # blurred = cv2.GaussianBlur(image, (5,5), 0)
        # sobel = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)[:1700, 1700:]
        # filtered = cv2.filter2D(sobel, -1, fil)
        # passes = int(np.log2(image.shape[1]))
        # store_lines = []
        # to_show = np.zeros_like(filtered)
        # prev_point = None
        # for col in range(0, filtered.shape[1], passes):
        #     checking_col = filtered[:,col]
        #     num_of_lines = 0
        #     for i in range(filtered.shape[0]):
        #         if checking_col[i] != 0:
        #             sucsess_bingo = 1
        #             for k in range(size + corr):
        #                 if i + k >= filtered.shape[0]:
        #                     continue
        #                 if (i, col, num_of_lines, sucsess_bingo) != prev_point:
        #                     if checking_col[i + k] !=0:
        #                         sucsess_bingo +=1
        #                     else:
        #                         if sucsess_bingo < size - corr//2:
        #                             break
        #                         else:
        #                             store_lines.append((i, col, num_of_lines, sucsess_bingo))
        #                             prev_point = (i, col, num_of_lines, sucsess_bingo)
        #                             num_of_lines += 1
        #                             break
        #                     if sucsess_bingo > size - corr//2:
        #                         store_lines.append((i, col, num_of_lines, sucsess_bingo))
        #                         prev_point = (i, col, num_of_lines, sucsess_bingo)
        #                         num_of_lines += 1
        #                         break
        #                 else:
        #                     break

        # for stored in store_lines:
        #     for i in range(stored[3] - 1):
        #         to_show[stored[0] + i][stored[1]] = 255
        # store_lines = []
        # for col in range(0, to_show.shape[1], passes):
        #     checking_col = to_show[:,col]
        #     maximum = size + 2*corr
        #     num_lines = 0
        #     for row in range(checking_col.shape[0]):
        #         if checking_col[row] != 0:
        #             iteration = 0
        #             while checking_col[row + iteration] !=0:
        #                 if row + iteration + 1 >= checking_col.shape[0]:
        #                     break
        #                 iteration += 1
                    
        #             if maximum >= iteration > size - corr:
        #                 iteration2 = 0
        #                 if row + iteration + 1 >= checking_col.shape[0]:
        #                     num_of_lines +=1
        #                     store_lines.append((row, col, iteration, num_lines))
        #                 else:
        #                     while checking_col[row + iteration + iteration2] == 0:
        #                         if row + iteration + iteration2 + 1 >= checking_col.shape[0]:
        #                             break
        #                         iteration2 += 1
        #                     if maximum >= iteration2 > size - corr:
        #                         num_of_lines +=1
        #                         store_lines.append((row, col, iteration, num_lines))
                                

        # to_show = np.zeros_like(filtered)
        # for stored in store_lines:
        #     for i in range(stored[2]):
        #         to_show[stored[0] + i][stored[1]] = 255
                
        # cv2.imshow("show", to_show)
        # cv2.waitKey(0)
    elif dim == "y":
        if inv:
            sets = 1
        else:
            sets = 0
        cloud = []
        edges = sm.Canny(image, 251, 255)
        edges = edges.astype(dtype=np.uint8)
        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                if edges[i][j] != 0 and edges[i][j] != 255:
                    edges[i][j] = 0
        
        patter = np.zeros_like(edges, dtype=np.uint16)
        last = patter.shape[1] % size
        for y in range(-int(angle*patter.shape[0] - 1), patter.shape[0] - last, size):
            for x in range(patter.shape[1]):
                    if ((x // size) + shift) % 2 == 0:
                        for k in range(size):
                            if 0<=int(angle * x) + y+k < patter.shape[0]:
                                patter[int(angle * x) + y+k][x] = 255

        patter_edges = sm.Canny(patter, 251, 255)
        cv2.imshow("image", image)
        cv2.imshow("ed", patter_edges)
        cv2.imshow("patt", patter.astype(dtype=np.uint8))
    else:
        return


full_light = cv2.imread("AgainMach/Photos/New/Good/FullLight.jpg")[560: 2400, 860:3800]
h,  w = full_light.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, camera_distortion, (w,h), 1, (w,h))
full_darkness = cv2.imread("AgainMach/Photos/New/Good/FullDarkness.jpg")[560: 2400, 860:3800]

full_light = cv2.undistort(full_light, camera_matrix, camera_distortion, None, newcameramtx)
full_darkness = cv2.undistort(full_darkness, camera_matrix, camera_distortion, None, newcameramtx)

full_light = cv2.warpPerspective(full_light, PerspectiveMatrix, (w, h))
full_darkness = cv2.warpPerspective(full_darkness, PerspectiveMatrix, (w, h))

vert_photo = cv2.imread("AgainMach/Photos/New/Good/VertSt.jpg")[560: 2400, 860:3800]
inv_vert_photo = cv2.imread("AgainMach/Photos/New/Good/VertInv.jpg")[560: 2400, 860:3800]
vert_photo = cv2.undistort(vert_photo, camera_matrix, camera_distortion, None, newcameramtx)
inv_vert_photo = cv2.undistort(inv_vert_photo, camera_matrix, camera_distortion, None, newcameramtx)

vert_photo = cv2.warpPerspective(vert_photo, PerspectiveMatrix, (w, h))
inv_vert_photo = cv2.warpPerspective(inv_vert_photo, PerspectiveMatrix, (w, h))

horiz_photo = cv2.imread("AgainMach/Photos/New/Good/HorSt.jpg")[560: 2400, 860:3800]
inv_horiz_photo = cv2.imread("AgainMach/Photos/New/Good/HorInv.jpg")[560: 2400, 860:3800]
horiz_photo = cv2.undistort(horiz_photo, camera_matrix, camera_distortion, None, newcameramtx)
inv_horiz_photo = cv2.undistort(inv_horiz_photo, camera_matrix, camera_distortion, None, newcameramtx)

horiz_photo = cv2.warpPerspective(horiz_photo, PerspectiveMatrix, (w, h))
inv_horiz_photo = cv2.warpPerspective(inv_horiz_photo, PerspectiveMatrix, (w, h))

gray_full_light = cv2.cvtColor(full_light, cv2.COLOR_BGR2GRAY)
gray_full_darkness = cv2.cvtColor(full_darkness, cv2.COLOR_BGR2GRAY)

gray_vert_photo = cv2.cvtColor(vert_photo, cv2.COLOR_BGR2GRAY)
gray_inv_vert_photo = cv2.cvtColor(inv_vert_photo, cv2.COLOR_BGR2GRAY)

gray_horiz_photo = cv2.cvtColor(horiz_photo, cv2.COLOR_BGR2GRAY)
gray_inv_horiz_photo = cv2.cvtColor(inv_horiz_photo, cv2.COLOR_BGR2GRAY)

l_d_v1_v2_h1_h2 = [full_light, full_darkness, vert_photo, inv_vert_photo, horiz_photo, inv_horiz_photo]
gray_l_d_v_vi_h_hi = [gray_full_light, gray_full_darkness, gray_vert_photo, gray_inv_vert_photo, gray_horiz_photo, gray_inv_horiz_photo]

if os.path.exists("AgainMach/Photos/New/Lited/LitedHor.jpg") and os.path.exists("AgainMach/Photos/New/Lited/LitedInvHor.jpg") and os.path.exists("AgainMach/Photos/New/Lited/LitedVert.jpg") and os.path.exists("AgainMach/Photos/New/Lited/LitedInvVert.jpg"):
    lited_pixels_vert = cv2.imread("AgainMach/Photos/New/Lited/LitedVert.jpg", cv2.IMREAD_GRAYSCALE)
    lited_pixel_inv_vert = cv2.imread("AgainMach/Photos/New/Lited/LitedInvVert.jpg", cv2.IMREAD_GRAYSCALE)

    lited_pixel_horiz = cv2.imread("AgainMach/Photos/New/Lited/LitedHor.jpg", cv2.IMREAD_GRAYSCALE)
    lited_pixel_inv_horiz = cv2.imread("AgainMach/Photos/New/Lited/LitedInvHor.jpg", cv2.IMREAD_GRAYSCALE)
else:
    print("Finding Ld_Lg")
    Ld_Lg = find_Ld_Lg_pixels(gray_l_d_v_vi_h_hi)
    print("Findind lited pixels vertical")
    lited_pixels_vert = find_Lited_pixels(gray_vert_photo, gray_inv_vert_photo, Ld_Lg)
    lited_pixel_inv_vert = find_Lited_pixels(gray_inv_vert_photo, gray_vert_photo, Ld_Lg)

    print("Findind lited pixels horizontal")
    lited_pixel_horiz = find_Lited_pixels(gray_horiz_photo, gray_inv_horiz_photo, Ld_Lg)
    lited_pixel_inv_horiz = find_Lited_pixels(gray_inv_horiz_photo, gray_horiz_photo, Ld_Lg)

    cv2.imwrite("AgainMach/Photos/New/Lited/LitedHor.jpg", lited_pixel_horiz)
    cv2.imwrite("AgainMach/Photos/New/Lited/LitedInvHor.jpg", lited_pixel_inv_horiz)
    cv2.imwrite("AgainMach/Photos/New/Lited/LitedVert.jpg", lited_pixels_vert)
    cv2.imwrite("AgainMach/Photos/New/Lited/LitedInvVert.jpg", lited_pixel_inv_vert)

print("Finding pattern")
line_width, pattern, shift, point_im, an = find_pattern(lited_pixel_horiz, lited_pixel_inv_horiz, dim="x")
pattern = pattern[:1700, 1700:]
if point_im:
    work_im = lited_pixel_inv_horiz[:1700, 1700:]
else:
    work_im = lited_pixel_horiz[:1700, 1700:]
print(f"Pattern size: {line_width}, Pattern shift: {shift}, Out image: {point_im}")

full_light, full_darkness, vert_photo, inv_vert_photo, horiz_photo, inv_horiz_photo = full_light[:1700, 1700:], full_darkness[:1700, 1700:], vert_photo[:1700, 1700:], inv_vert_photo[:1700, 1700:], horiz_photo[:1700, 1700:], inv_horiz_photo[:1700, 1700:]
gray_full_light, gray_full_darkness, gray_vert_photo, gray_inv_vert_photo, gray_horiz_photo, gray_inv_horiz_photo = gray_full_light[:1700, 1700:], gray_full_darkness[:1700, 1700:], gray_vert_photo[:1700, 1700:], gray_inv_vert_photo[:1700, 1700:], gray_horiz_photo[:1700, 1700:], gray_inv_horiz_photo[:1700, 1700:]

print("Finding curves")
cloud_ = find_curves(work_im, line_width, shift, an, zazor=11)

cloud2_ = find_curves(lited_pixel_horiz[:1700, 1700:], line_width, shift, an, inv=True, zazor=11)
final_cloud = []
for item in cloud_:
    final_cloud.append(item)
for item in cloud2_:
    final_cloud.append(item)

final_cloud = np.array(final_cloud, dtype=np.float64)
final_cloud = filter_cloud(final_cloud)
cv2.waitKey(0)
# print("Finding point clouds")
# point_cloud_vert = get_point_cloud(lited_pixel_inv_vert, lited_pixel_inv_vert, camera_matrix, distance=4)
# point_cloud_horiz = get_point_cloud(lited_pixel_horiz, lited_pixel_inv_horiz, camera_matrix, distance=4)
# print(len(point_cloud_vert), len(point_cloud_horiz))
# print("Filtering clouds")
# final_point_cloud = filter_cloud(point_cloud_horiz, point_cloud_vert, distance=1)

# with open('ASC_POINTS_3.asc', 'w') as f:
#     for point in final_cloud:
#         f.write(f"{point[0]} {point[1]} {point[2]}\n")

with open('AgainMach/Points/PCD_POINTS_FLOAT_MEAN_45_x5.pcd', 'w') as f:
    f.write("# .PCD v0.7 - Point Cloud Data file format\n")
    f.write("VERSION 0.7\n")
    f.write("FIELDS x y z\n")
    f.write("SIZE 4 4 4\n")
    f.write("TYPE F F F\n")
    f.write("COUNT 1 1 1\n")
    f.write(f"WIDTH {len(cloud_)}\n")
    f.write("HEIGHT 1\n")
    f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
    f.write(f"POINTS {len(cloud_)}\n")
    f.write("DATA ascii\n")

    for point in final_cloud:
        f.write(f"{point[0]} {point[1]} {point[2]}\n")

# with open('PLY_POINTS_3.ply', 'w') as f:
#     f.write("ply\n")
#     f.write("format ascii 1.0\n")
#     f.write(f"element vertex {len(cloud_)}\n")
#     f.write("property float x\n")
#     f.write("property float y\n")
#     f.write("property float z\n")
#     f.write("end_header\n")

#     for point in final_cloud:
#         f.write(f"{point[0]} {point[1]} {point[2]}\n")



x = final_cloud[:, 0]
y = final_cloud[:, 1]
z = final_cloud[:, 2]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(x, y, z)

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()
kk=5
max_x = np.int32(np.max(x))
max_y = np.int32(np.max(y))
depth_image = np.zeros(shape=(max_y + 1, max_x + 1, 3), dtype=np.uint8)
for point in final_cloud:
    if 0 <= point[2] < 5*kk:
        depth_image[int(point[1])][int(point[0])] = [0, 0, 0]
    elif 5*kk <= point[2] < 10*kk:
        depth_image[int(point[1])][int(point[0])] = [255, 0, 0]
    elif 10*kk <= point[2] < 15*kk:
        depth_image[int(point[1])][int(point[0])] = [100, 0, 0]
    elif 15*kk <= point[2] < 20*kk:
        depth_image[int(point[1])][int(point[0])] = [14, 118, 237]
    elif 20*kk <= point[2] < 25*kk:
        depth_image[int(point[1])][int(point[0])] = [96, 164, 244]
    elif 25*kk <= point[2] < 30*kk:
        depth_image[int(point[1])][int(point[0])] = [0, 0, 100]
    elif 30*kk <= point[2] < 35*kk:
         depth_image[int(point[1])][int(point[0])] = [0, 0, 255]
    elif 35*kk <= point[2]:
        depth_image[int(point[1])][int(point[0])] = [255, 255, 255]

# cv2.imshow("depth", depth_image)

cv2.waitKey(0)

cv2.imwrite("AgainMach/Photos/New/Out/Depth40_11_x5.jpg", depth_image)
