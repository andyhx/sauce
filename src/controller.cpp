#include "controller.h"

void Controller::show_usage() {
  cout << "usage:" << endl;
  cout << "./sauce command" << endl;
  cout << "command = extract | train | test" << endl;
}

void Controller::generate(char* input, char* output, int width, int height, int h_stride, int v_stride) {
    vector<string> dirs = Controller::listdir(input);
    int n=0;
    for(string& s : dirs) {
      Mat image = imread(s);
      Mat g;
      Sobel(image, g, -1, 1, 1);
      g = abs(g);

      int best = 0;
      int x=0, y=0;
      for(int i=0; i<g.rows-height; i+=v_stride) {
        for(int j=0; j<g.cols-width; j+=h_stride) {
          Mat windowH = g.rowRange(i, i+height);
          Mat window = windowH.colRange(j, j+width);
          Scalar windowSum = sum(window);
          if(windowSum(0) > best) {
            best = windowSum(0);
            x = j;
            y = i;
          }
        }
      }

      Mat finalWindowH = image.rowRange(y, y+height);
      Mat finalWindow = finalWindowH.colRange(x, x+width);
      char buf[1024];
      sprintf(buf, "%s/%d.png", output, n++);
      imwrite(buf, finalWindow);
    }
}
  
void Controller::extract(Descriptor* desc, char* dir, char* output) {
    Mat features(0, 0, CV_32FC1);
    vector<string> dirs = Controller::listdir(dir);
    for(string& s : dirs) {
      Acc acc;
      acc.m = imread(s);
      Acc resultAcc = desc->iterate(acc);
      Controller::addFeature(features, resultAcc.m);
    }
    FileStorage fs(output, FileStorage::WRITE);
    fs << "features" << features;
}

void Controller::train(char* pos, char* neg, char* output) {
    FileStorage positives(pos, FileStorage::READ);
    FileStorage negatives(neg, FileStorage::READ);
    Mat posM; positives["features"] >> posM;
    Mat negM; negatives["features"] >> negM;
    Mat features(posM.rows+negM.rows, posM.cols, posM.type());
    Mat labels = Mat::zeros(posM.rows+negM.rows, 1, CV_32FC1);

    for(int i=0; i<posM.rows; i++) {
      posM.row(i).copyTo(features.row(i));
      labels.at<float>(i, 0) = 1;
    }
    for(int j=0; j<negM.rows; j++) {
      negM.row(j).copyTo(features.row(posM.rows+j));
    }

    CvSVM svm; CvSVMParams params;
    params.kernel_type = CvSVM::LINEAR;
    svm.train(features, labels, Mat(), Mat(), params);
    svm.save(output);
}

void Controller::predict(Descriptor* desc, char* set, char* model) {
  CvSVM svm;
  svm.load(model);

  int zeros=0, ones=0;
  vector<string> dirs = listdir(set);
  for(string& s : dirs) {
    Acc acc;
    acc.m = imread(s);
    Acc resultAcc = desc->iterate(acc);
    int predict = svm.predict(resultAcc.m.t());
    if(predict == 0) {
      zeros++;
    }
    else {
      ones++;
    }
  }
  int total = ones+zeros;
  cout << "Zeros: " << zeros << " (" << (float)(zeros)/total << ")" << endl;
  cout << "Ones: " << ones << " (" <<  (float)ones/total << ")" << endl;
  cout << "Total: " << total << endl;
};

void Controller::detect(char* input, char* annotations) {
  vector<BOX> boxes = pascal(annotations);
  Mat image = imread(input);
  for(BOX& b : boxes) {
    rectangle(image, Rect(get<0>(b), get<1>(b), get<2>(b)-get<0>(b), get<3>(b)-get<1>(b)),
              Scalar(0, 0, 255), 2);
  }
  imshow("im", image);
  waitKey();
};

auto Controller::pascal(char* input) -> vector<BOX> {
  ifstream file;
  file.open(input);

  vector<string> goodLines;
  string line;
  if(file.is_open()) {
    int n = 0;
    while(file.good()) {
      getline(file, line);
      if(line!="" && line[0]!='#' && n++ > 3) {
        goodLines.push_back(line);
      }
    }
  }

  vector<BOX> boxes;
  for(int i=0; i < goodLines.size(); i+=3) {
    string box = goodLines[i+2];
    size_t found = box.find_first_of(":", 0);
    string coords = box.substr(found+2);
    
    size_t firstComma = coords.find_first_of(",", 0);
    size_t secondComma = coords.find_first_of(",", firstComma+1);
    size_t firstRight = coords.find_first_of(")", 0);
    size_t secondRight = coords.find_first_of(")", firstRight+1);
    size_t secondLeft = coords.find_first_of("(", 1);

    string xmin = coords.substr(1, firstComma-1);
    string ymin = coords.substr(firstComma+2, firstRight-firstComma-2);
    string xmax = coords.substr(secondLeft+1, secondComma-secondLeft-1);
    string ymax = coords.substr(secondComma+2, secondRight-secondComma-2); 
    
    boxes.push_back(BOX(atoi(xmin.c_str()), atoi(ymin.c_str()), atoi(xmax.c_str()), atoi(ymax.c_str())));
  }

  return boxes;
};

auto Controller::addFeature(Mat& features, Mat s) -> void {
  if(s.rows == 0 && s.cols == 0) {
    features = s;
  }
  else {
    features.push_back(s);
  }
};

auto Controller::listdir(const char* path) -> vector<string>{
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
