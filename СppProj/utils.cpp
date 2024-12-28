#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include "utils.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;



void PrintMat(Mat mat) {
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            if (mat.type() == CV_32F) {
                std::cout << mat.at<float>(i, j) << " ";
            }
            else if (mat.type() == CV_8U) {
                std::cout << static_cast<int>(mat.at<uchar>(i, j)) << " ";
            }
            else if (mat.type() == CV_8UC3) {
                cv::Vec3b pixel = mat.at<cv::Vec3b>(i, j);
                std::cout << "(" << static_cast<int>(pixel[0]) << ", "<< static_cast<int>(pixel[1]) << ", "<< static_cast<int>(pixel[2]) << ") ";
            }
            else if (mat.type() == CV_32S) {
                std::cout << mat.at<int>(i, j) << " ";
            }
        }
        std::cout << std::endl;
    }
}

float GaussFunc(float x, float y, float sigma) {
    return (1. / sqrt(2. * M_PI * pow(sigma, 2))) * exp(-(pow(x, 2) + pow(y, 2)) / (2. * pow(sigma, 2)));
}

Mat GetGaussian(int shape, float sigma) {
    Mat blank_array = Mat::zeros(Size(shape, shape), CV_32F);
    for (int i = -(shape / 2); i <= shape / 2; i++) {
        for (int j = -(shape / 2); j <= shape / 2; j++) {
            blank_array.at<float>(i + shape/2, j + shape/2) = GaussFunc(i, j, sigma);
        }
    }
    float sum = 0;
    for (int i = 0; i < shape; i++) {
        for (int j = 0; j < shape; j++) {
            sum += blank_array.at<float>(i, j);
        }
    }
    if(sum != 0) {
        for (int i = 0; i < shape; i++) {
            for (int j = 0; j < shape; j++) {
                blank_array.at<float>(i, j) /= sum;
            }
        }
    }
    return blank_array;
}

Mat MakeBorder(cv::Mat image, cv::Size size, BorderType border) {
    int h = image.rows;
    int w = image.cols;
    int y_center = size.height;
    int x_center = size.width;
    Mat padded;
    if (image.channels() == 3) {
        padded = Mat::zeros(Size(w + x_center * 2, h+y_center*2), CV_8UC3);
    }
    else {
        padded = Mat::zeros(Size(w + x_center * 2, h + y_center * 2), CV_8UC1);
    }
    image.copyTo(padded(Rect(x_center, y_center, w, h)));
    switch (border)
    {
    case ZERO_BORDER:
        break;
    case REPLICATE_BORDER:
        if (image.channels() == 3) {
            for (int i = 0; i < y_center; i++) {
                for (int j = x_center; j < w + x_center; j++) {
                    padded.at<Vec3b>(i, j) = padded.at<Vec3b>(y_center, j);
                }
            }
            for (int i = h+1; i < h+y_center+1; i++) {
                for (int j = x_center; j < w + x_center; j++) {
                    padded.at<Vec3b>(i, j) = padded.at<Vec3b>(h, j);
                }
            }
            for (int i = y_center; i < h+1; i++) {
                for (int j = 0; j < x_center; j++) {
                    padded.at<Vec3b>(i, j) = padded.at<Vec3b>(i, x_center);
                }
            }
            for (int i = y_center; i < h+1; i++) {
                for (int j = w+1; j < w+x_center+1; j++) {
                    padded.at<Vec3b>(i, j) = padded.at<Vec3b>(i, w);
                }
            }
        }
        else {
            for (int i = 0; i < y_center; i++) {
                for (int j = x_center; j < w + x_center; j++) {
                    padded.at<uchar>(i, j) = padded.at<uchar>(y_center, j);
                }
            }
            for (int i = h + 1; i < h + y_center + 1; i++) {
                for (int j = x_center; j < w + x_center; j++) {
                    padded.at<uchar>(i, j) = padded.at<uchar>(h, j);
                }
            }
            for (int i = y_center; i < h + 1; i++) {
                for (int j = 0; j < x_center; j++) {
                    padded.at<uchar>(i, j) = padded.at<uchar>(i, x_center);
                }
            }
            for (int i = y_center; i < h + 1; i++) {
                for (int j = w + 1; j < w + x_center + 1; j++) {
                    padded.at<uchar>(i, j) = padded.at<uchar>(i, w);
                }
            }
        }
        break;
    case REFLECT_BORDER:
        if (image.channels() == 3) {
            for (int i = 0; i < y_center; i++) {
                for (int j = x_center; j < w + x_center; j++) {
                    padded.at<Vec3b>(i, j) = padded.at<Vec3b>(y_center + i, j);
                }
            }
            for (int i = h + 1; i < h + y_center + 1; i++) {
                for (int j = x_center; j < w + x_center; j++) {
                    padded.at<Vec3b>(i, j) = padded.at<Vec3b>(h-i, j);
                }
            }
            for (int i = y_center; i < h + 1; i++) {
                for (int j = 0; j < x_center; j++) {
                    padded.at<Vec3b>(i, j) = padded.at<Vec3b>(i, x_center+j);
                }
            }
            for (int i = y_center; i < h + 1; i++) {
                for (int j = w + 1; j < w + x_center + 1; j++) {
                    padded.at<Vec3b>(i, j) = padded.at<Vec3b>(i, w-j);
                }
            }
        }
        else {
            for (int i = 0; i < y_center; i++) {
                for (int j = x_center; j < w + x_center; j++) {
                    padded.at<uchar>(i, j) = padded.at<uchar>(y_center + i, j);
                }
            }
            for (int i = h + 1; i < h + y_center + 1; i++) {
                for (int j = x_center; j < w + x_center; j++) {
                    padded.at<uchar>(i, j) = padded.at<uchar>(h-i, j);
                }
            }
            for (int i = y_center; i < h + 1; i++) {
                for (int j = 0; j < x_center; j++) {
                    padded.at<uchar>(i, j) = padded.at<uchar>(i, x_center+j);
                }
            }
            for (int i = y_center; i < h + 1; i++) {
                for (int j = w + 1; j < w + x_center + 1; j++) {
                    padded.at<uchar>(i, j) = padded.at<uchar>(i, w-j);
                }
            }
        }
        break;
    case PERIOD_BORDER:
        if (image.channels() == 3) {
            for (int i = 0; i < y_center; i++) {
                for (int j = x_center; j < w + x_center; j++) {
                    padded.at<Vec3b>(i, j) = padded.at<Vec3b>(h-i, j);
                }
            }
            for (int i = h + 1; i < h + y_center + 1; i++) {
                for (int j = x_center; j < w + x_center; j++) {
                    padded.at<Vec3b>(i, j) = padded.at<Vec3b>(y_center + i - h + 1, j);
                }
            }
            for (int i = y_center; i < h + 1; i++) {
                for (int j = 0; j < x_center; j++) {
                    padded.at<Vec3b>(i, j) = padded.at<Vec3b>(i, w-j);
                }
            }
            for (int i = y_center; i < h + 1; i++) {
                for (int j = w + 1; j < w + x_center + 1; j++) {
                    padded.at<Vec3b>(i, j) = padded.at<Vec3b>(i, x_center + j - w - 1);
                }
            }
        }
        else {
            for (int i = 0; i < y_center; i++) {
                for (int j = x_center; j < w + x_center; j++) {
                    padded.at<uchar>(i, j) = padded.at<uchar>(h - i, j);
                }
            }
            for (int i = h + 1; i < h + y_center + 1; i++) {
                for (int j = x_center; j < w + x_center; j++) {
                    padded.at<uchar>(i, j) = padded.at<uchar>(y_center + i - h + 1, j);
                }
            }
            for (int i = y_center; i < h + 1; i++) {
                for (int j = 0; j < x_center; j++) {
                    padded.at<uchar>(i, j) = padded.at<uchar>(i, w - j);
                }
            }
            for (int i = y_center; i < h + 1; i++) {
                for (int j = w + 1; j < w + x_center + 1; j++) {
                    padded.at<uchar>(i, j) = padded.at<uchar>(i, x_center + j - w - 1);
                }
            }
        }
        break;
    case REFLECT_BORDER_101:
        if (image.channels() == 3) {
            for (int i = 0; i < y_center; i++) {
                for (int j = x_center; j < w + x_center; j++) {
                    padded.at<Vec3b>(i, j) = padded.at<Vec3b>(y_center + i + 1, j);
                }
            }
            for (int i = h + 1; i < h + y_center + 1; i++) {
                for (int j = x_center; j < w + x_center; j++) {
                    padded.at<Vec3b>(i, j) = padded.at<Vec3b>(h - i - 1, j);
                }
            }
            for (int i = y_center; i < h + 1; i++) {
                for (int j = 0; j < x_center; j++) {
                    padded.at<Vec3b>(i, j) = padded.at<Vec3b>(i, x_center + j + 1);
                }
            }
            for (int i = y_center; i < h + 1; i++) {
                for (int j = w + 1; j < w + x_center + 1; j++) {
                    padded.at<Vec3b>(i, j) = padded.at<Vec3b>(i, w - j - 1);
                }
            }
        }
        else {
            for (int i = 0; i < y_center; i++) {
                for (int j = x_center; j < w + x_center; j++) {
                    padded.at<uchar>(i, j) = padded.at<uchar>(y_center + i + 1, j);
                }
            }
            for (int i = h + 1; i < h + y_center + 1; i++) {
                for (int j = x_center; j < w + x_center; j++) {
                    padded.at<uchar>(i, j) = padded.at<uchar>(h - i - 1, j);
                }
            }
            for (int i = y_center; i < h + 1; i++) {
                for (int j = 0; j < x_center; j++) {
                    padded.at<uchar>(i, j) = padded.at<uchar>(i, x_center + j + 1);
                }
            }
            for (int i = y_center; i < h + 1; i++) {
                for (int j = w + 1; j < w + x_center + 1; j++) {
                    padded.at<uchar>(i, j) = padded.at<uchar>(i, w - j - 1);
                }
            }
        }
        break;
    default:
        break;
    }
    return padded;
}

Mat ApplyFilter(Mat image, Mat kernel, BorderType border, bool cross) {
    Mat blank_image(image.size(), image.type());
    int m = kernel.rows;
    int n = kernel.cols;
    int y_center = m / 2;
    int x_center = n / 2;
    int h = image.rows;
    int w = image.cols;
    Mat padded = MakeBorder(image, Size(y_center, x_center), border);
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            Rect region_rect(x, y, n, m);
            Mat region = padded(region_rect);
            if (image.channels() == 3) {
                int r = 0, g = 0, b = 0;
                if (!cross) {
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < n; j++) {
                            b += int(region.at<Vec3b>(i, j)[0]) * kernel.at<float>(i, j);
                            g += int(region.at<Vec3b>(i, j)[1]) * kernel.at<float>(i, j);
                            r += int(region.at<Vec3b>(i, j)[2]) * kernel.at<float>(i, j);
                        }
                    }
                }
                else {
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < n; j++) {
                            b += int(region.at<Vec3b>(i, j)[0]) * kernel.at<int>(i, j);
                            g += int(region.at<Vec3b>(i, j)[1]) * kernel.at<int>(i, j);
                            r += int(region.at<Vec3b>(i, j)[2]) * kernel.at<int>(i, j);
                        }
                    }
                }
               /* if (y % 10 == 0) {
                    cout << "(" << r << ", " << g << ", " << b << ")" << endl;
                }*/
                r = std::min(std::max(r, 0), 255);
                g = std::min(std::max(g, 0), 255);
                b = std::min(std::max(b, 0), 255);
                blank_image.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
            }
            else {
                int s = 0;
                if (!cross) {
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < n; j++) {
                            s += region.at<uchar>(i, j) * kernel.at<float>(i, j);
                        }
                    }
                }
                else {
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < n; j++) {
                            s += region.at<uchar>(i, j) * kernel.at<int>(i, j);
                        }
                    }
                }

                s = std::min(std::max(s, 0), 255);
                blank_image.at<uchar>(y, x) = static_cast<uchar>(s);
            }
        }
    }
    return blank_image;
}

Mat Derivative(Mat image, char dim, BorderType border) {
    vector<int> x_sobel = {-1, 0, 1,-2, 0, 2, -1, 0, 1};
    vector<int> y_sobel = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    Mat sobel_x(3, 3, CV_32S, x_sobel.data());
    Mat sobel_y(3, 3, CV_32S, y_sobel.data());
    Mat buff;
    switch (dim) {
    case 'x':
        return ApplyFilter(image, sobel_x, border, true);
        break;
    case 'y':
        return ApplyFilter(image, sobel_y, border, true);
        break;
    case '0':
        buff = ApplyFilter(image, sobel_x, border, true);
        return ApplyFilter(buff, sobel_y, border, true);
        break;
    default:
        break;
    }
}

Mat CannyEdge(Mat image, int weak, int strong, bool retStrong, bool ret255) {
    Mat gauss_55 = GetGaussian(5, 1.4);
    Mat gray_img;
    cvtColor(image, gray_img, COLOR_BGR2GRAY);
    Mat filtered = ApplyFilter(gray_img, gauss_55);
    
    Mat Der_x = Derivative(gray_img, 'x');
    imshow("fil", Der_x);
    Mat Der_y = Derivative(gray_img, 'y');
    Mat angles = Mat::zeros(gray_img.size(), CV_64F);
    Mat modules = Mat::zeros(gray_img.size(), CV_64F);
    for (int i = 0; i < filtered.rows;i++) {
        for (int j = 0; j < filtered.cols; j++) {
            modules.at<double>(i, j) = sqrt(pow(double(Der_x.at<uchar>(i,j)), 2) + pow(double(Der_y.at<uchar>(i, j)), 2));
            if (Der_x.at<uchar>(i, j) != 0) {
                angles.at <double> (i, j) = (atan(double(Der_y.at<uchar>(i, j)) / double(Der_x.at<uchar>(i, j))) * 180.0 / M_PI);
            }
            else {
                angles.at<double>(i, j) = 90.0f;
            }
            /*cout << "(" << int(Der_x.at<uchar>(i, j)) << ", " << angles.at<double>(i, j) << ")" << endl;*/
        }
    }
    double max_mod;
    minMaxLoc(modules, nullptr, &max_mod);
    if (isnan(weak)) {
        weak = 0.2 * max_mod;
    }
    if (isnan(strong)) {
        strong = 0.5 * max_mod;
    }
    //cout << max_mod;
    Mat output = Mat::zeros(modules.size(), CV_8UC1);
    for (int x = 0; x < modules.cols;x++) {
        for (int y = 0; y < modules.rows;y++) {
            double grad_angle = angles.at<double>(y, x);
            grad_angle = abs(grad_angle) > 180 ? abs(grad_angle - 180) : abs(grad_angle);
            int neighb_1_x, neighb_1_y, neighb_2_x, neighb_2_y;
            if (grad_angle <= 22.5) {
                neighb_1_x = x - 1; neighb_1_y = y;
                neighb_2_x = x + 1; neighb_2_y = y;
            }
            else if (22.5 < grad_angle && grad_angle <= 22.5 + 45.) {
                neighb_1_x = x - 1; neighb_1_y = y - 1;
                neighb_2_x = x + 1; neighb_2_y = y + 1;
            }
            else if (45. + 22.5 < grad_angle && grad_angle <= 90. + 22.5) {
                neighb_1_x = x; neighb_1_y = y - 1;
                neighb_2_x = x; neighb_2_y = y + 1;
            }
            else if (90. + 22.5 < grad_angle && grad_angle <= 135. + 22.5) {
                neighb_1_x = x - 1; neighb_1_y = y + 1;
                neighb_2_x = x + 1; neighb_2_y = y - 1;
            }
            else if (135. + 22.5 < grad_angle && grad_angle <= 180.0 + 22.5) {
                neighb_1_x = x - 1; neighb_1_y = y;
                neighb_2_x = x + 1; neighb_2_y = y;
            }
            if (neighb_1_x >= 0 && neighb_1_x < modules.cols && neighb_1_y >= 0 && neighb_1_y < modules.rows) {
                if (modules.at<double>(y, x) < modules.at<double>(neighb_1_y, neighb_1_x)) {
                    modules.at<double>(y,x) = 0;
                    continue;
                }
            }

            if (neighb_2_x >= 0 && neighb_2_x < modules.cols && neighb_2_y >= 0 && neighb_2_y < modules.rows) {
                if (modules.at<double>(y, x) < modules.at<double>(neighb_2_y, neighb_2_x)) {
                    modules.at<double>(y, x) = 0;
                    continue;
                }
            }
            double grad_mod = modules.at<double>(y, x);
            if (grad_mod < weak) {
                output.at<uchar>(y, x) = 0;
            }
            else if (weak <= grad_mod && grad_mod <= strong) {
                if (retStrong) {
                    output.at<uchar>(y, x) = 0;
                }
                else {
                    output.at<uchar>(y, x) = ret255 ? 255 : uint8_t(grad_mod);
                }
            }
            else {
                output.at<uchar>(y, x) = ret255 ? 255 : uint8_t(grad_mod);
            }
        }
    }
    return output;
}
