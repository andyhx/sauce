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
      Mat histogram = Mat::zeros(1, 512, CV_32FC1);

      for(int ii=0; ii<block.rows-2; ii++) {
        for(int jj=0; jj<block.cols-2; jj++) {
          Mat kernel = block.rowRange(ii, ii+3).colRange(jj, jj+3);
          Mat output = (Mat_<unsigned char>(8,1) << kernel.at<unsigned char>(0,0), kernel.at<unsigned char>(0,1), kernel.at<unsigned char>(0,2),
              kernel.at<unsigned char>(1,2), kernel.at<unsigned char>(2,2), kernel.at<unsigned char>(2,1),
              kernel.at<unsigned char>(2,0), kernel.at<unsigned char>(1,0));

          int value = 0;
          for(int iii=0;iii<output.rows;iii++) {
            if(abs(output.at<unsigned char>(iii,0) - kernel.at<unsigned char>(1,1)) > threshold) {
              value = (value << 1) | 1;
            }
            else {
              value = (value << 1);
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
