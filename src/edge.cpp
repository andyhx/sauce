#include "edge.h"

int Edge::n = 3;
Mat Edge::edgelets = Mat::zeros(10, 9, CV_8UC1);

auto Edge::convert_to_grayscale(Acc a) -> Acc {
    Mat gray_image;
    cvtColor(a.m, gray_image, CV_RGB2GRAY);
    a.m = gray_image;
    return a;
};

auto Edge::calculate_gradient(Acc a) -> Acc {
    Mat grad = (Mat_<char>(3,1) << 1, 0, -1);
    Mat input(grad.rows, grad.cols, CV_16SC1);
    a.m.convertTo(input, CV_16SC1);
    Mat gradient_x;
    Mat gradient_y;
    filter2D(input, gradient_x, -1, grad);
    filter2D(input, gradient_y, -1, grad.t());
    tuple<Mat, Mat> t2(gradient_x, gradient_y);
    a.t2 = t2;
    return a;
};

auto Edge::calculate_magnitude_orientation(Acc a) -> Acc {
    Mat x = get<0>(a.t2);
    Mat y = get<1>(a.t2);
    Mat magnitude(x.rows, x.cols, CV_32FC1);
    Mat orientation(x.rows, x.cols, CV_8UC1);
    for(int i=0; i<x.rows; i++) {
        for(int j=0; j<x.cols; j++) {
          float vx = (float)x.at<short int>(i,j);
          float vy = (float)y.at<short int>(i,j);
          magnitude.at<float>(i,j) = sqrt(vx*vx + vy*vy);
          float orient = atan2(vy,vx);
          if(orient < 0) {
            orient += CV_PI;
          }
          int bin = round((orient*6)/CV_PI)+1;
          if(bin == 7) bin = 1;
          orientation.at<unsigned char>(i,j) = bin;
        }
    }
    a.t2 = tuple<Mat, Mat>(magnitude, orientation);
    return a;
}

auto Edge::calculate_affinities(Acc a) -> Acc {
    float L[] = {1, 0.8, 0.5, 0, 0.5, 0.8};
    Mat magnitude = get<0>(a.t2);
    Mat orientation = get<1>(a.t2);
    Mat affinities = Mat::zeros(1, edgelets.rows, CV_32FC1);

    for(int i=0; i<magnitude.rows-n+1; i+=n) {
      for(int j=0; j<magnitude.cols-n+1; j+=n) {
        Mat ma = magnitude.rowRange(i, i+n).colRange(j, j+n);
        Mat oa = orientation.rowRange(i, i+n).colRange(j, j+n);
        Mat mac; ma.copyTo(mac);
        Mat oac; oa.copyTo(oac);
        Mat m = mac.reshape(0,1);
        Mat o = oac.reshape(0,1);

        for(int ii=0; ii<edgelets.rows; ii++) {
            float aff = 0;
            for(int jj=0; jj<edgelets.cols; jj++) {
              if(edgelets.at<unsigned char>(ii,jj) > 0) {
                aff += m.at<float>(0, jj) * L[abs(o.at<unsigned char>(0, jj)-edgelets.at<unsigned char>(ii,jj))+1];
              }     
            }
            affinities.at<float>(0, ii) += aff/edgelets.rows;
        }
      }
    }
    a.m = affinities;
    return a;
};

vector<function<Acc(Acc)>> Edge::features() {
    vector<function<Acc(Acc)>> funs;
    funs.push_back(convert_to_grayscale); 
    funs.push_back(calculate_gradient);
    funs.push_back(calculate_magnitude_orientation);
    funs.push_back(calculate_affinities);
    return funs; 
};
