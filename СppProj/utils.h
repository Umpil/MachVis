#pragma once
#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
enum BorderType {
    NO_BORDER,
    ZERO_BORDER,
    REPLICATE_BORDER,
    REFLECT_BORDER,
    PERIOD_BORDER,
    REFLECT_BORDER_101
};

float GaussFunc(float x, float y, float sigma);
cv::Mat GetGaussian(int shape, float sigma);
void PrintMat(cv::Mat);

cv::Mat ApplyFilter(cv::Mat image, cv::Mat kernel, BorderType border=ZERO_BORDER, bool cross=false);
cv::Mat MakeBorder(cv::Mat image, cv::Size size, BorderType border=REPLICATE_BORDER);
cv::Mat Derivative(cv::Mat, char dim = 'x', BorderType border=REPLICATE_BORDER);
cv::Mat CannyEdge(cv::Mat image, int weak=NAN, int strong=NAN, bool retStrong=true, bool ret255=true);
#endif 