#include "hog.h"

int main(int argc, char** argv) {
  char* imageName = argv[1];

  Mat image;
  image = imread(imageName, 1);

  Acc acc;
  acc.m = image;
  for(function<Acc&(Acc&)>& fun : HOG::steps()) {
      acc = fun(acc);
  }
  tuple<Mat, Mat> t = acc.t2;
  imshow("Gradient x", get<0>(t));
  imshow("Gradient y", get<1>(t));

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
    Mat gradient_x(s.height, s.width, CV_16SC1);
    Mat gradient_y(s.height, s.width, CV_16SC1);
    filter2D(a.m, gradient_x, -1, grad);
    filter2D(a.m, gradient_y, -1, grad.t());
    tuple<Mat, Mat> t2(gradient_x, gradient_y);
    a.t2 = t2;
    return a;
};

vector<function<Acc&(Acc&)>> HOG::steps() {
    vector<function<Acc&(Acc&)>> funs;
    funs.push_back(convert_to_grayscale); 
    funs.push_back(calculate_gradient);
    return funs; 
};
