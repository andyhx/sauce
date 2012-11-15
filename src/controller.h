#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "global.h"
#include "descriptor.h"

class Controller {
  static auto addFeature(Mat& features, Mat s) -> void;
  static auto listdir(const char* path) -> vector<string>;
  public:
  static void show_usage();
  static void extract(Descriptor* desc, char* dir, char* output);
  static void train(char* pos, char* neg, char* output);
  static void predict(Descriptor* desc, char* set, char* model);
};

#endif
