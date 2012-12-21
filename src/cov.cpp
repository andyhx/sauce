#include "cov.h"

int Cov::blockWidth = 16;
int Cov::hStride = 16;
int Cov::vStride = 16;

auto Cov::convert_to_grayscale(Acc a) -> Acc {
    Mat gray_image;
    cvtColor(a.m, gray_image, CV_RGB2GRAY);
    a.m = gray_image;
    return a;
};


auto Cov::calculate_covariances(Acc a) -> Acc {
    Mat im = a.m;
    Mat image;
    im.convertTo(image, CV_32FC1);
    int covarianceMatrices = (image.rows / blockWidth) * (image.cols / blockWidth);
    Mat covs(0, 0, CV_32FC1);

    for(int i=0; i<image.rows-blockWidth+1; i+=vStride) {
      for(int j=0; j<image.cols-blockWidth+1; j+=hStride) {
          Mat block = image.rowRange(i, i+blockWidth).colRange(j, j+blockWidth);
          Mat Ix, Iy, Ixx, Iyy;
          Sobel(block, Ix, -1, 1, 0);
          Sobel(block, Iy, -1, 0, 1);
          Sobel(block, Ixx, -1, 2, 0);
          Sobel(block, Iyy, -1, 0, 2);
          Ix = abs(Ix);
          Iy = abs(Iy);
          Ixx = abs(Ixx);
          Iyy = abs(Iyy);

          Mat magnitude(Ix.rows, Ix.cols, CV_32FC1);
          Mat orientation(Ix.rows, Ix.cols, CV_32FC1);
          for(int ii=0; ii<Ix.rows; ii++) {
            for(int jj=0; jj<Ix.cols; jj++) {
              float vx = Ix.at<float>(ii,jj);
              float vy = Iy.at<float>(ii,jj);
              magnitude.at<float>(ii,jj) = sqrt(vx*vx + vy*vy);
              orientation.at<float>(ii,jj) = atan2(vy,vx);
            }
          }

          Mat features(Ix.rows*Ix.cols, 6, CV_32FC1);
          features.col(0) = Ix.reshape(0, 1).t();
          features.col(1) = Iy.reshape(0, 1).t();
          features.col(2) = Ixx.reshape(0, 1).t();
          features.col(3) = Iyy.reshape(0, 1).t();
          features.col(4) = magnitude.reshape(0, 1).t();
          features.col(5) = orientation.reshape(0, 1).t();

          Mat covariance, mean;
          calcCovarMatrix(features, covariance, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
          covs.push_back(covariance.reshape(0, 1));
      }
    }
    covs.convertTo(a.m, CV_32FC1);
    return a;
};

auto Cov::normalize_blocks(Acc a) -> Acc {
    Mat bins = a.m;
    for(int i=0; i<bins.rows; i++) {
      Mat row = bins.rowRange(i, i+1);
      row /= (norm(row, NORM_L2) + 0.001);
    }
    return a;
}

auto Cov::flatten_features(Acc a) -> Acc {
    a.m = a.m.reshape(0, 1);
    return a;
}

vector<function<Acc(Acc)>> Cov::features() {
    vector<function<Acc(Acc)>> funs;
    funs.push_back(convert_to_grayscale); 
    funs.push_back(calculate_covariances);
    funs.push_back(normalize_blocks);
    funs.push_back(flatten_features);
    return funs; 
};
