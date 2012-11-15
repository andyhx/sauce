#include "hog.h"
#include "controller.h"

int main(int argc, char** argv) {

  auto listdir = [](const char *path) {
    DIR *pdir = NULL;
    pdir = opendir(path);
    struct dirent *pent = NULL;

    vector<string> files;
    while(pent = readdir(pdir)) {
      string file(pent->d_name);
      if(file!="." && file!="..") {
        file = "/"+file;
        files.push_back(path + file);
      }
    }
    closedir(pdir);
    return files;
  };

  if(argc <= 1) {
    Controller::show_usage();
    return 0;
  }
  if(strcmp("extract", argv[1]) == 0) {
    cout << "extract" << endl;
  }
  else if(strcmp("train", argv[1]) == 0) {
    cout << "train" << endl;
  }
  else if(strcmp("test", argv[1]) == 0) {
    cout << "test" << endl;
  }
  else {
    Controller::show_usage();
  }

  /*Mat features(0, 0, CV_32FC1);
  Mat labels(0, 0, CV_32FC1);
  auto addFeature = [&](Mat sample) {
    Size s = features.size();
    if(s.width == 0 && s.height == 0) {
      features = sample;
    }
    else {
      features.push_back(sample);
    }
  };

  vector<string> dirs = listdir("/media/FC1A11C21A117B3A/inz/priv/sanity_test/train/pos");
  for(string& s : dirs) {
    Acc acc;
    acc.m = imread(s);
    Acc resultAcc = HOG::iterate(HOG::features(), acc);
    addFeature(resultAcc.m);
    cout << s << endl;
  }
  Mat labelsPos(dirs.size(), 1, CV_32FC1, Scalar::all(1));
  labels.push_back(labelsPos);

  dirs = listdir("/media/FC1A11C21A117B3A/inz/priv/sanity_test/train/neg");
  for(string& s : dirs) {
    Acc acc;
    acc.m = imread(s);
    Acc resultAcc = HOG::iterate(HOG::features(), acc);
    addFeature(resultAcc.m);
    cout << s << endl;
  }
  Mat labelsNeg(dirs.size(), 1, CV_32FC1, Scalar::all(0));
  labels.push_back(labelsNeg);

  CvSVM svm;
  CvSVMParams params;
  params.kernel_type = CvSVM::LINEAR;
  svm.train(features, labels, Mat(), Mat(), params);
  svm.save("hogtrain");*/

  /*  CvSVM svm;
      svm.load("hogtrain");

      int zeros=0, ones=0;
      vector<string> dirs = listdir("/media/FC1A11C21A117B3A/inz/priv/sanity_test/test/neg");
      for(string& s : dirs) {
      Acc acc;
      acc.m = imread(s);
      Acc resultAcc = HOG::iterate(HOG::features(), acc);
      int predict = svm.predict(resultAcc.m.t());
      cout << s << " " << predict << endl;

      if(predict == 0) {
      zeros++;
      }
      else {
      ones++;
      }
      }

      cout << "Zeros: " << zeros << endl;
      cout << "Ones: " << ones << endl;
      cout << "Total: " << ones+zeros << endl;*/

  return 0;
}


