#include "controller.h"

void Controller::show_usage() {
  cout << "usage:" << endl;
  cout << "./sauce command" << endl;
  cout << "command = extract | train | test" << endl;
}

void Controller::generate(char* input, char* output, int width, int height, int h_stride, int v_stride, int nrand) {
    vector<string> dirs = Controller::listdir(input);
    srand(time(NULL));

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
      sprintf(buf, "%s/sobel_%03d.png", output, n);
      imwrite(buf, finalWindow);

      for(int k=0; k<nrand; k++) {
        int i = rand()%(image.rows-height);
        int j = rand()%(image.cols-width);
        Mat window = image.rowRange(i, i+height).colRange(j, j+width);
        sprintf(buf, "%s/rand_%03d_%03d.png", output, n, k);
        imwrite(buf, window);
      }

      n++;
    }
}
  
void Controller::extract(Descriptor* desc, char* dir, char* output, float probability) {
    Mat features(0, 0, CV_32FC1);
    vector<string> dirs = Controller::listdir(dir);
    srand(time(NULL));
    for(string& s : dirs) {
      if(probability == 1.0) {
        Mat image = imread(s);
        Controller::add_feature(features, extract_features(desc, image));
      }
      else {
        float prob = (float)rand()/INT_MAX;
        if(prob < probability) {
          Mat image = imread(s);
          Controller::add_feature(features, extract_features(desc, image));
        }
      }
    }
    FileStorage fs(output, FileStorage::WRITE);
    fs << "features" << features;
}

void Controller::join_sets(char* a, char* b, char* output) {
    FileStorage featuresA(a, FileStorage::READ);
    FileStorage featuresB(b, FileStorage::READ);
    Mat aM; featuresA["features"] >> aM;
    Mat bM; featuresB["features"] >> bM;

    for(int i=0; i<bM.rows; i++) {
      aM.push_back(bM.rowRange(i, i+1));
    }

    FileStorage fs(output, FileStorage::WRITE);
    fs << "features" << aM;
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
};

RESULT Controller::predict(Descriptor* desc, char* set, char* model) {
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
  return RESULT(zeros, ones, total);
};

void Controller::detect(Descriptor* desc, char* model, char* input, char* annotations, char* output, int width, int height, int h_stride, int v_stride) {
  CvSVM svm;
  svm.load(model);

  vector<string> images = listdir(input);
  map<int, pair<int, int>> detectionRate;
  ofstream outputFile;
  outputFile.open(output, ios::out);

  for(int ii=0; ii<images.size(); ii++) {
    int falsePositives = 0, truePositives = 0, falseSamples = 0;
    float scales[] = {0.25, 0.5, 0.75, 1};

    Mat image = imread(images[ii]);
    Mat original;
    float scale = 600./image.rows;
    resize(image, image, Size(0,0), scale, scale);
    image.copyTo(original);

    vector<BOX> boxes = pascal(annotation_file(annotations, images[ii]), scale);
    vector<BOX> detections;
    for(float &s : scales) {
      Mat scaled;
      resize(original, scaled, Size(0,0), s, s);
      for(int i=0; i<scaled.rows-height+1; i+=v_stride) {
        for(int j=0; j<scaled.cols-width+1; j+=h_stride) {
          Mat window = scaled.rowRange(i, i+height).colRange(j, j+width);
          Mat sample = extract_features(desc, window);
          int predict = svm.predict(sample);

          if(is_false_positive(BOX(j/s, i/s, (j+width)/s, (i+height)/s), boxes)) {
            falseSamples++;
            if(predict == 1)
              falsePositives++;
          }

          if(predict == 1) {
            detections.push_back(BOX(j/s, i/s, (j+width)/s, (i+height)/s));
          }
        }
      }
    }

    //for(BOX& b : boxes) {
      //rectangle(image, Rect(ELEMENT(0,b), ELEMENT(1,b), ELEMENT(2,b)-ELEMENT(0,b), ELEMENT(3,b)-ELEMENT(1,b)),
          //Scalar(0, 0, 255), 2);
    //}
    //for(BOX& b : detections) {
      //rectangle(image, Rect(ELEMENT(0,b), ELEMENT(1,b), ELEMENT(2,b)-ELEMENT(0,b), ELEMENT(3,b)-ELEMENT(1,b)),
          //is_false_positive(b, boxes) ? Scalar(255,0,0) : Scalar(0, 255, 0), 2);
    //}

    //char buf[1024];
    //sprintf(buf, "/media/FC1A11C21A117B3A/inz/priv/det_cov4/%d.png", ii);
    //imwrite(buf, image);
    int trues = count_true_positives(detections, boxes);
    if(falsePositives > 0) {
      int base = round(log10((float)falsePositives/falseSamples)); 
      map<int, pair<int, int>>::iterator it = detectionRate.find(base);
      if(it == detectionRate.end()) {
          detectionRate[base] = pair<int, int>(trues, boxes.size());
      }
      else {
          pair<int, int> det = detectionRate[base];
          detectionRate[base] = pair<int, int>(det.first + trues, det.second + boxes.size());
      }
      outputFile << (float)falsePositives/falseSamples << " " << trues << " " << boxes.size() << " " << endl; 
    }
  }

  for(pair<const int, pair<int, int>>& detection : detectionRate) {
    outputFile << detection.first << " " << 1 - (float)(detection.second.first)/detection.second.second << endl;
  }
};

void Controller::false_positives(Descriptor* desc, char* model, char* input, char* output, int width, int height, int h_stride, int v_stride) {
  CvSVM svm;
  svm.load(model);

  vector<string> images = listdir(input);
  int n = 0;

  for(string& s : images) {
    Mat image = imread(s);
    float scale = 600./image.rows;
    resize(image, image, Size(0,0), scale, scale);

    float scales[] = {0.5, 0.75, 1};
    for(float &s : scales) {
      vector<BOX> detections;
      Mat scaled;
      resize(image, scaled, Size(0,0), s, s);

      for(int i=0; i<scaled.rows-height+1; i+=v_stride) {
        for(int j=0; j<scaled.cols-width+1; j+=h_stride) {
          Mat window = scaled.rowRange(i, i+height).colRange(j, j+width);
          Mat sample = extract_features(desc, window);
          int predict = svm.predict(sample);
          if(predict == 1) {
            BOX detection(j/s, i/s, (j+width)/s, (i+height)/s);
            char buf[1024];
            sprintf(buf, "%s/fp_%04d.png", output, n++);
            imwrite(buf, window);
          }
        }
      }
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
    if(MAX(ELEMENT(0,b), ELEMENT(0,detection)) < MIN(ELEMENT(2,b), ELEMENT(2,detection)) &&
       MAX(ELEMENT(1,b), ELEMENT(1,detection)) < MIN(ELEMENT(3,b), ELEMENT(3,detection)))
    {
      //check if they overlap on at least 50% of smaller one's area
      int x1 = MAX(ELEMENT(0,b), ELEMENT(0,detection));
      int x2 = MIN(ELEMENT(2,b), ELEMENT(2,detection));
      int y1 = MAX(ELEMENT(1,b), ELEMENT(1,detection));
      int y2 = MIN(ELEMENT(3,b), ELEMENT(3,detection));
      int width = x2-x1;
      int height = y2-y1;
      int boxArea = (ELEMENT(2,b) - ELEMENT(0,b)) * (ELEMENT(3,b) - ELEMENT(1,b));
      int detectionArea = (ELEMENT(2,detection) - ELEMENT(0,detection)) * (ELEMENT(3,detection)-ELEMENT(1,detection));

      if(3*width*height >= (boxArea > detectionArea ? detectionArea : boxArea))
        return false;
    }  
  }
  return true;
};

auto Controller::count_true_positives(vector<BOX>& detections, vector<BOX>& boxes) -> int {
  int dets = 0;
  for(BOX& b : boxes) {
    for(BOX& detection : detections) {
      //check if detected window and bounding box overlap
      if(MAX(ELEMENT(0,b), ELEMENT(0,detection)) < MIN(ELEMENT(2,b), ELEMENT(2,detection)) &&
          MAX(ELEMENT(1,b), ELEMENT(1,detection)) < MIN(ELEMENT(3,b), ELEMENT(3,detection)))
      {
        //check if they overlap on at least 50% of smaller one's area
        int x1 = MAX(ELEMENT(0,b), ELEMENT(0,detection));
        int x2 = MIN(ELEMENT(2,b), ELEMENT(2,detection));
        int y1 = MAX(ELEMENT(1,b), ELEMENT(1,detection));
        int y2 = MIN(ELEMENT(3,b), ELEMENT(3,detection));
        int width = x2-x1;
        int height = y2-y1;
        int boxArea = (ELEMENT(2,b) - ELEMENT(0,b)) * (ELEMENT(3,b) - ELEMENT(1,b));
        int detectionArea = (ELEMENT(2,detection) - ELEMENT(0,detection)) * (ELEMENT(3,detection)-ELEMENT(1,detection));

        if(2*width*height >= (boxArea > detectionArea ? detectionArea : boxArea)) {
          dets++;
          break;
        } 
      }
    }
  }
  return dets;
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
  
auto Controller::annotation_file(char* annotations, string file) -> string {
  size_t found = file.find_last_of("/\\");
  string fileName = file.substr(found+1);
  found = fileName.find_first_of(".");
  string name = fileName.substr(0, found);
  char buf[1024];
  sprintf(buf, "%s/%s.txt", annotations, name.c_str());
  return string(buf);
};

