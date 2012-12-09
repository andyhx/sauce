#ifndef GLOBAL_H
#define GLOBAL_H
#include <opencv2/opencv.hpp>
#include <functional>
#include <iostream>
#include <fstream>
#include <new>
#include <dirent.h>
#include <unistd.h>
#include <limits.h>

using namespace std;
using namespace cv; 

struct Acc {
  Mat m;
  tuple<Mat, Mat> t2;
};

typedef tuple<int, int, int, int> BOX;
#define ELEMENT(N, TUPLE) get<N>(TUPLE)


#endif
