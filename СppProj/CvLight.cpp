#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <iostream>
using namespace cv;
using namespace std;

int main()
{
    Mat image = imread("C:/Users/User/Documents/CV/Enot.jpg");
    Mat edges = CannyEdge(image, 100, 200, false);
    imshow("edges", edges);
    waitKey(0);
}




