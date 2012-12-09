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
          Mat window = g.rowRange(i, i+height).colRange(j, j+width);
          Scalar windowSum = sum(window);
          if(windowSum(0) > best) {
            best = windowSum(0);
            x = j;
            y = i;
          }
        }
      }

      Mat finalWindow = image.rowRange(y, y+height).colRange(x, x+width);
      char buf[1024];
      sprintf(buf, "%s/%d.png", output, n++);
      imwrite(buf, finalWindow);
    }
}
  
void Controller::extract(Descriptor* desc, char* dir, char* output) {
    Mat features(0, 0, CV_32FC1);
    vector<string> dirs = Controller::listdir(dir);
    for(string& s : dirs) {
      Mat image = imread(s);
      Controller::add_feature(features, extract_features(desc, image));
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
    Mat result = extract_features(desc, imread(s));
    int predict = svm.predict(result);
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

void Controller::detect(Descriptor* desc, char* model, char* input, char* annotations) {
  CvSVM svm;
  svm.load(model);

  int width = 64;
  int height = 128;
  int h_stride = 32;
  int v_stride = 64;

  vector<string> images = listdir(input);
  vector<string> annots = listdir(annotations);

  for(int ii=0; ii<images.size(); ii++) {
    Mat image = imread(images[ii]);
    Mat original;
    float scale = 450./image.rows;
    resize(image, image, Size(0,0), scale, scale);
    image.copyTo(original);

    vector<BOX> boxes = pascal(annots[ii], scale);
    for(BOX& b : boxes) {
      rectangle(image, Rect(ELEMENT(0,b), ELEMENT(1,b), ELEMENT(2,b)-ELEMENT(0,b), ELEMENT(3,b)-ELEMENT(1,b)),
          Scalar(0, 0, 255), 2);
    }

    float scales[] = {0.3, 0.5, 0.6, 0.75, 1, 1.15};

    for(float &s : scales) {
      vector<BOX> detections;
      Mat scaled;
      resize(original, scaled, Size(0,0), s, s);
      for(int i=0; i<scaled.rows-height; i+=v_stride) {
        for(int j=0; j<scaled.cols-width; j+=h_stride) {
          Mat window = scaled.rowRange(i, i+height).colRange(j, j+width);
          Mat sample = extract_features(desc, window);
          int predict = svm.predict(sample);
          if(predict == 1) {
            detections.push_back(BOX(j/scale, i/scale, (j+width)/scale, (i+height)/scale));
          }
        }
      }

      Mat newImage;
      image.copyTo(newImage);
      for(BOX& b : detections) {
        rectangle(newImage, Rect(ELEMENT(0,b), ELEMENT(1,b), ELEMENT(2,b)-ELEMENT(0,b), ELEMENT(3,b)-ELEMENT(1,b)),
            is_false_positive(b, boxes) ? Scalar(255,0,0) : Scalar(0, 255, 0), 2);
      }

      char buf[1024];
      sprintf(buf, "/media/FC1A11C21A117B3A/inz/priv/det_cov3/%d_%f.png", ii, s);
      imwrite(buf, newImage);
    }
  }
};

auto Controller::pascal(string input, float scale) -> vector<BOX> {
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
    
    boxes.push_back(BOX(atoi(xmin.c_str())*scale, atoi(ymin.c_str())*scale, atoi(xmax.c_str())*scale, atoi(ymax.c_str())*scale));
  }

  return boxes;
};

auto Controller::extract_features(Descriptor* desc, Mat image) -> Mat {
  if(image.rows == 160 && image.cols == 96) {
    image = image.rowRange(16,128+16).colRange(16,64+16);
  }
  else if(image.rows == 134 && image.cols == 70) {
    image = image.rowRange(3,128+3).colRange(3,64+3);
  }
  Acc acc;
  acc.m = image;
  Acc resultAcc = desc->iterate(acc);
  return resultAcc.m;
};

auto Controller::add_feature(Mat& features, Mat s) -> void {
  if(s.rows == 0 && s.cols == 0) {
    features = s;
  }
  else {
    features.push_back(s);
  }
};

auto Controller::is_false_positive(BOX detection, vector<BOX>& boxes) -> bool {
  for(BOX& b : boxes) {
    //check if detected window and bounding box overlap
    if(MAX(get<0>(b), get<0>(detection)) < MIN(get<2>(b), get<2>(detection)) &&
       MAX(get<1>(b), get<1>(detection)) < MIN(get<3>(b), get<3>(detection)))
    {
      //check if they overlap on at least 50% of smaller one's area
      int x1 = MAX(get<0>(b), get<0>(detection));
      int x2 = MIN(get<2>(b), get<2>(detection));
      int y1 = MAX(get<1>(b), get<1>(detection));
      int y2 = MIN(get<3>(b), get<3>(detection));
      int width = x2-x1;
      int height = y2-y1;
      int boxArea = (get<2>(b) - get<0>(b)) * (get<3>(b) - get<1>(b));
      int detectionArea = (get<2>(detection) - get<0>(detection)) * (get<3>(detection) - get<1>(detection));

      if(2*width*height >= (boxArea > detectionArea ? detectionArea : boxArea))
        return false;
    }  
  }
  return true;
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
