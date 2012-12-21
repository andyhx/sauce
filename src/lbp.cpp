#include "lbp.h"

int LBP::mask = 3;
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
  for(int i=0; i<image.rows-mask+1; i++) {
    for(int j=0; j<image.cols-mask+1; j++) {
      Mat kernel = image.rowRange(i, i+3).colRange(j, j+3);
      int centerVal = kernel.at<unsigned char>(1, 1);

      Mat output = (Mat_<unsigned char>(8,1) << kernel.at<unsigned char>(0,0), kernel.at<unsigned char>(0,1), kernel.at<unsigned char>(0,2),
        kernel.at<unsigned char>(1,2), kernel.at<unsigned char>(2,2), kernel.at<unsigned char>(2,1),
        kernel.at<unsigned char>(2,0), kernel.at<unsigned char>(1,0));

      for(int ii=0; ii<output.rows; ii++) {
        if(abs(output.at<unsigned char>(ii, 0) - centerVal) > threshold) {
          output.at<unsigned char>(ii, 0) = 1;
        }
        else {
          output.at<unsigned char>(ii, 0) = 0;
        }
      }

      if(features.rows == 0) {
        features = output.t();
      }
      else {
        Mat feature = output.t();
        features.push_back(feature);
      }
    }
  }
  a.m = features;  
  return a;
};

auto LBP::perform_binning(Acc a) -> Acc {
    Mat features = a.m;
    Mat histogram = Mat::zeros(features.cols, features.cols, CV_32FC1);

    for(int i=0; i<features.rows; i++) {
      Mat feature = features.rowRange(i, i+1);

      int insideArc = 0;
      int arcStart = 0;
      Mat arcs(0, 0, CV_8UC1);
      
      for(int j=0; j<=features.cols; j++) {
        if(j==features.cols && insideArc) {
          insideArc = 0;
          Mat arc = (Mat_<unsigned char>(1,2) << arcStart, j);
          arcs.push_back(arc);
          break;
        }
        if(feature.at<unsigned char>(0, j) == 1 && insideArc == 0){
          arcStart = j+1;
          insideArc = 1;
        }
        else if(feature.at<unsigned char>(0,j) == 0 && insideArc == 1) {
          insideArc = 0;
          Mat arc = (Mat_<unsigned char>(1,2) << arcStart, j);
          arcs.push_back(arc);
        }
      }
      int angle = 0;
      int length = 0;
      if(arcs.rows == 2 && arcs.at<unsigned char>(0,0) == 1 && arcs.at<unsigned char>(1,1) == features.cols) {
          angle = arcs.at<unsigned char>(1,0);
          length = arcs.at<unsigned char>(1,1)-angle+1+arcs.at<unsigned char>(0,1);
      }
      else if(arcs.rows == 1) {
          angle = arcs.at<unsigned char>(0,0);
          length = arcs.at<unsigned char>(0,1)-angle+1;
      }

      if(angle>0 && length>0) {
        histogram.at<float>(angle-1, length-1)++;
      }

    }

    a.m = histogram;
    return a;
};

auto LBP::normalize_blocks(Acc a) -> Acc {
    Mat bins = a.m;
    for(int i=0; i<bins.rows; i++) {
      Mat row = bins.rowRange(i, i+1);
      row /= (norm(row, NORM_L2) + 0.001);
    }

    return a;
}

auto LBP::flatten_features(Acc a) -> Acc {
    a.m = a.m.reshape(0, 1);
    return a;
}

vector<function<Acc(Acc)>> LBP::features() {
    vector<function<Acc(Acc)>> funs;
    funs.push_back(convert_to_grayscale); 
    funs.push_back(threshold_masks);
    funs.push_back(perform_binning);
    funs.push_back(normalize_blocks);
    funs.push_back(flatten_features);
    return funs; 
};
