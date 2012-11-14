#include <opencv2/opencv.hpp>
#include <functional>
#include <iostream>
#include <new>

using namespace std;
using namespace cv; 

struct Acc {
  Mat m;
  tuple<Mat, Mat> t2;
};
