#include "hog.h"

int main(int argc, char** argv) {
  char* imageName = argv[1];

  Mat image;
  image = imread(imageName, 1);

  imshow("Image", image);
  Acc acc;
  acc.m = image;
  for(function<Acc&(Acc&)>& fun : HOG::steps()) {
      acc = fun(acc);
  }
  tuple<Mat, Mat> t = acc.t2;
  imshow("Gradient x", get<0>(t));
  imshow("Gradient y", get<1>(t));

  imshow("HOG", acc.m);
  waitKey(0);
  return 0;
}


auto HOG::convert_to_grayscale(Acc& a) -> Acc& {
    Mat gray_image;
    cvtColor(a.m, gray_image, CV_RGB2GRAY);
    a.m = gray_image;
    return a;
};

auto HOG::calculate_gradient(Acc& a) -> Acc& {
    Mat grad = (Mat_<char>(1,3) << -1, 0, 1);
    Size s = a.m.size();
    Mat input(s.height, s.width, CV_16SC1);
    a.m.convertTo(input, CV_16SC1);
    Mat gradient_x;
    Mat gradient_y;
    filter2D(input, gradient_x, -1, grad);
    filter2D(input, gradient_y, -1, grad.t());
    tuple<Mat, Mat> t2(gradient_x, gradient_y);
    a.t2 = t2;
    return a;
};

auto HOG::calculate_magnitude_orientation(Acc& a) -> Acc& {
    Mat x = get<0>(a.t2);
    Mat y = get<1>(a.t2);
    Size s = x.size();
    Mat magnitude(s.height, s.width, CV_8UC1);
    Mat orientation(s.height, s.width, CV_8UC1);
    for(int i=0; i<s.height; i++) {
        for(int j=0; j<s.width; j++) {
          Point p(j,i);
          double vx = x.at<short int>(p);
          double vy = y.at<short int>(p);
          magnitude.at<uchar>(p) = (uchar)sqrt(vx*vx + vy*vy);
          int orient = (int)(atan2(vx, vy)*180/CV_PI);
          orient = (orient < 0 ? 360+orient : orient);
          orient = (orient > 180 ? orient-180 : orient);
          orientation.at<uchar>(p) = (uchar)orient;
        }
    }
    a.t2 = tuple<Mat, Mat>(magnitude, orientation);
    return a;
}

auto HOG::perform_binning(Acc& a) -> Acc& {
    int cellWidth = 3;
    int bins = 10;
    float binSize = 180/(bins+0.0);
    Mat magnitude = get<0>(a.t2);
    Mat orientation = get<1>(a.t2);
    Size s = magnitude.size();
    Mat cells((s.width/cellWidth)*(s.height/cellWidth), bins, CV_64FC1, Scalar::all(0));
    for(int i=0; i<s.height; i++) {
      for(int j=0; j<s.width; j++) {
        // cell numbers on x and y axes, start from 1, left top corner
        int cellX = ceil((j-0.5*cellWidth)/cellWidth);
        int cellY = ceil((i-0.5*cellWidth)/cellWidth);
        // coordinates inside cells, start from 1
        int xCell = (j+1)-(cellX-1)*cellWidth;
        int yCell = (i+1)-(cellY-1)*cellWidth;
        // coordinates of neighbouring cells centers, start from 1
        // (x0, y0) - top left corner cell center
        // (x1, y1) - bottom right corner cell center
        int x0 = round((cellX - 0.5)*cellWidth);
        int x1 = x0+cellWidth;
        int y0 = round((cellY - 0.5)*cellWidth);
        int y1 = y0+cellWidth;
        
        if(x0 > 0 && y0 > 0 && y1 <= s.height && x1 <= s.width) {
          tuple<int, int> xes[] = {tuple<int, int>(x0, x1), tuple<int, int>(x1, x0)};
          tuple<int, int> yes[] = {tuple<int, int>(y0, y1), tuple<int, int>(y1, y0)};
          for(tuple<int, int> &x : xes) {
            for(tuple<int, int> &y : yes) {
              int x_0 = get<0>(x);
              int x_1 = get<1>(x);
              int y_0 = get<0>(y);
              int y_1 = get<1>(y);
              int row = (y_0/cellWidth) * (s.width/cellWidth) + (x_0/cellWidth);
              int col = orientation.at<uchar>(i,j) / binSize;
              double xc = (j+1-x_0)/(x_1-x_0+0.0);
              double yc = (i+1-y_0)/(y_1-y_0+0.0);
              double interpolation = xc * yc * magnitude.at<uchar>(i,j);
              cells.at<double>(row, col) += interpolation;
            }
          }
        }
      }
    }
    a.m = cells;
    return a;
}

auto HOG::normalize_blocks(Acc& a) -> Acc& {
    int blockWidth = 3;
    Mat histograms = a.m;
    Size s = histograms.size();
    int height = s.height/blockWidth;
    Mat blocks(0, s.width*blockWidth, CV_64FC1, Scalar::all(0));
    for(int i=0; i<height; i++) {
      Mat block = histograms.rowRange(0, blockWidth);
      Mat blockV = block.reshape(0,1);
      Mat normalized = blockV/(norm(blockV, NORM_L1)+0.0001);
      blocks.push_back(normalized);
      s = histograms.size();
      if(s.height >= 2*blockWidth) {
        histograms = histograms.rowRange(blockWidth, s.height);
      }
    }
    a.m = blocks;
    return a;
}

vector<function<Acc&(Acc&)>> HOG::steps() {
    vector<function<Acc&(Acc&)>> funs;
    funs.push_back(convert_to_grayscale); 
    funs.push_back(calculate_gradient);
    funs.push_back(calculate_magnitude_orientation);
    funs.push_back(perform_binning);
    funs.push_back(normalize_blocks);
    return funs; 
};
