#include "lbp.h"

int LBP::blockWidth = 16;
int LBP::threshold = 30;

auto LBP::convert_to_grayscale(Acc a) -> Acc {
    Mat gray_image;
    cvtColor(a.m, gray_image, CV_RGB2GRAY);
    a.m = gray_image;
    return a;
};

auto LBP::threshold_masks(Acc a) -> Acc {
  Mat image = a.m;

  Mat features;
  for(int i=0; i<image.rows-2; i+=3) {
    for(int j=0; j<image.cols-2; j+=3) {
      Mat kernel = image.rowRange(i, i+3).colRange(j, j+3);

      int value = 0;
      for(int ii=0;ii<kernel.rows;ii++) {
        for(int jj=0;jj<kernel.cols;jj++) {
          if(abs((int)kernel.at<unsigned char>(ii,jj) - kernel.at<unsigned char>(1,1)) > threshold) {
            value = (value << 1) | 1;
          }
          else {
            value = (value << 1);
          }
        }
      }
      Mat f = Mat_<float>(1,1) << (float)value;
      features.push_back(f);
    }
  }
  a.m = features.t();  
  return a;
};


vector<function<Acc(Acc)>> LBP::features() {
    vector<function<Acc(Acc)>> funs;
    funs.push_back(convert_to_grayscale); 
    funs.push_back(threshold_masks);
    return funs; 
};
