#include "lbp.h"

int LBP::blockWidth = 32;
int LBP::threshold = 30;

auto LBP::convert_to_grayscale(Acc a) -> Acc {
    Mat gray_image;
    cvtColor(a.m, gray_image, CV_RGB2GRAY);
    a.m = gray_image;
    return a;
};

auto LBP::threshold_masks(Acc a) -> Acc {
  Mat image = a.m;

  int vStride = blockWidth;
  int hStride = blockWidth;

  Mat histograms;
  for(int i=0; i<image.rows-blockWidth+1; i+=vStride) {
    for(int j=0; j<image.cols-blockWidth+1; j+=hStride) {
      Mat block = image.rowRange(i, i+blockWidth).colRange(j, j+blockWidth);
      Mat histogram = Mat::zeros(1, 1023, CV_32FC1);

      for(int ii=0; ii<block.rows-2; ii++) {
        for(int jj=0; jj<block.cols-2; jj++) {
          Mat kernel = block.rowRange(ii, ii+3).colRange(jj, jj+3);

          int value = 0;
          for(int iii=0;iii<kernel.rows;iii++) {
            for(int jjj=0;jjj<kernel.cols;jjj++) {
              if(abs(kernel.at<unsigned char>(iii,jjj) - kernel.at<unsigned char>(1,1)) > threshold) {
                value = (value << 1) | 1;
              }
              else {
                value = (value << 1);
              }
            }
          }
          histogram.at<float>(0, value)++;
        }
      }
      histogram /= norm(histogram, NORM_L2) + 0.001;
      histograms.push_back(histogram);
    }
  }
  a.m = histograms; 
  return a;
};

auto LBP::flatten_features(Acc a) -> Acc {
    a.m = a.m.reshape(0, 1);
    return a;
};

vector<function<Acc(Acc)>> LBP::features() {
    vector<function<Acc(Acc)>> funs;
    funs.push_back(convert_to_grayscale); 
    funs.push_back(threshold_masks);
    funs.push_back(flatten_features);
    return funs; 
};
