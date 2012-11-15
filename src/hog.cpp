#include "hog.h"

auto HOG::convert_to_grayscale(Acc a) -> Acc {
    Mat gray_image;
    cvtColor(a.m, gray_image, CV_RGB2GRAY);
    a.m = gray_image;
    return a;
};

auto HOG::calculate_gradient(Acc a) -> Acc {
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

auto HOG::calculate_magnitude_orientation(Acc a) -> Acc {
    Mat x = get<0>(a.t2);
    Mat y = get<1>(a.t2);
    Mat magnitude(x.rows, x.cols, CV_32FC1);
    Mat orientation(x.rows, x.cols, CV_32FC1);
    for(int i=0; i<x.rows; i++) {
        for(int j=0; j<x.cols; j++) {
          float vx = (float)x.at<short int>(i,j);
          float vy = (float)y.at<short int>(i,j);
          magnitude.at<float>(i,j) = sqrt(vx*vx + vy*vy);
          float orient = atan2(vy,vx);
          if(orient < 0) {
            orient += CV_PI;
          }
          orientation.at<float>(i,j) = orient;
        }
    }
    a.t2 = tuple<Mat, Mat>(magnitude, orientation);
    return a;
}

auto HOG::perform_binning(Acc a) -> Acc {
    int cellWidth = 8;
    int binsNo = 9;

    Mat mag = get<0>(a.t2);
    Mat orient = get<1>(a.t2);
    int xCells = mag.cols / cellWidth;
    int yCells = mag.rows / cellWidth;

    Mat angleLut = Mat::zeros(binsNo, 1, CV_32FC1);
    Mat xLut = Mat::zeros(xCells, 1, CV_32FC1);
    Mat yLut = Mat::zeros(yCells, 1, CV_32FC1);
    Mat bins = Mat::zeros(xCells*yCells, binsNo, CV_32FC1);

    for(int i=0; i<binsNo; i++) {
      angleLut.at<float>(i, 0) = (CV_PI/binsNo) * (i+0.5);
    }
    for(int i=0; i<xCells; i++) {
      xLut.at<float>(i, 0) = (float)mag.cols/xCells * (i+0.5);
    }
    for(int i=0; i<yCells; i++) {
      yLut.at<float>(i, 0) = (float)mag.rows/yCells * (i+0.5);
    }

    for(int i=0; i<mag.rows; i++) {
      for(int j=0; j<mag.cols; j++) {
        float angle = orient.at<float>(i,j);
        int bin = round(angle/CV_PI * binsNo);

        int angleBin2, angleBin1;
        float ca1, ca2;
        if(bin == 0) {
          angleBin2 = 0;
          angleBin1 = 0;
          ca1 = 0;
          ca2 = 1;
        }
        else if(bin == binsNo) {
          angleBin2 = binsNo-1;
          angleBin1 = binsNo-1;
          ca1 = 0;
          ca2 = 1;
        }
        else {
          angleBin2 = bin;
          angleBin1 = bin-1;
          float angle2 = angleLut.at<float>(angleBin2, 0);
          float angle1 = angleLut.at<float>(angleBin1, 0);
          ca1 = (angle-angle1)/(angle2-angle1);
          ca2 = (angle2-angle)/(angle2-angle1);
        }

        int x = j;
        int xBin = round(((float)x/mag.cols) * xCells); 
        int xBin2, xBin1;
        float cax1, cax2;
        if(xBin == 0) {
          xBin2 = 0;
          xBin1 = 0;
          cax1 = 0;
          cax2 = 1;
        }
        else if(xBin == xCells) {
          xBin2 = xCells-1;
          xBin1 = xCells-1;
          cax1 = 0;
          cax2 = 1;
        }
        else {
          xBin2 = xBin;
          xBin1 = xBin-1;
          float x2 = xLut.at<float>(xBin2, 0);
          float x1 = xLut.at<float>(xBin1, 0);
          cax1 = (x-x1)/(x2-x1);
          cax2 = (x2-x)/(x2-x1);
        }

        int y = i;
        int yBin = round(((float)y/mag.rows) * yCells); 
        int yBin1, yBin2;
        float cay1, cay2;
        if(yBin == 0) {
          yBin1 = 0;
          yBin2 = 0;
          cay1 = 0;
          cay2 = 1;
        }
        else if(yBin == yCells) {
          yBin1 = yCells-1;
          yBin2 = yCells-1;
          cay1 = 0;
          cay2 = 1;
        }
        else {
          yBin2 = yBin;
          yBin1 = yBin-1;
          float y2 = yLut.at<float>(yBin2, 0);
          float y1 = yLut.at<float>(yBin1, 0);
          cay1 = (y-y1)/(y2-y1);
          cay2 = (y2-y)/(y2-y1);
        }

        int h1 = yBin1 * xCells + xBin1;
        int h2 = yBin1 * xCells + xBin2;
        int h3 = yBin2 * xCells + xBin1;
        int h4 = yBin2 * xCells + xBin2;
        float m = mag.at<float>(i,j);

        bins.at<float>(h1, angleBin1) += cax1*cay1*ca1*m;
        bins.at<float>(h1, angleBin2) += cax1*cay1*ca2*m;
        bins.at<float>(h2, angleBin1) += cax2*cay1*ca1*m;
        bins.at<float>(h2, angleBin2) += cax2*cay1*ca2*m;
        bins.at<float>(h3, angleBin1) += cax1*cay2*ca1*m;
        bins.at<float>(h3, angleBin2) += cax1*cay2*ca2*m;
        bins.at<float>(h4, angleBin1) += cax2*cay2*ca1*m;
        bins.at<float>(h4, angleBin2) += cax2*cay2*ca2*m;
      }
    }
    a.m = bins;
    return a;
}

auto HOG::normalize_blocks(Acc a) -> Acc {
    int blockWidth = 3;
    Mat bins = a.m;
    //for(int i=0; i<bins.rows; i++) {
      //Mat row = bins.rowRange(i, i+1);
      //row = row / (norm(row, NORM_L2) + 0.001);
    //}
    bins /= (norm(bins, NORM_L2) + 0.0001);
    return a;
}

auto HOG::flatten_features(Acc a) -> Acc {
    a.m = a.m.reshape(0, 1);
    return a;
}

vector<function<Acc(Acc)>> HOG::features() {
    vector<function<Acc(Acc)>> funs;
    funs.push_back(convert_to_grayscale); 
    funs.push_back(calculate_gradient);
    funs.push_back(calculate_magnitude_orientation);
    funs.push_back(perform_binning);
    funs.push_back(normalize_blocks);
    funs.push_back(flatten_features);
    return funs; 
};
