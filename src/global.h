#ifndef GLOBAL_H
#define GLOBAL_H
#include <opencv2/opencv.hpp>
#include <functional>
#include <iostream>
#include <new>
#include <dirent.h>
#include <unistd.h>

using namespace std;
using namespace cv; 

struct Acc {
  Mat m;
  tuple<Mat, Mat> t2;
};
#endif
