#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "global.h"
#include "descriptor.h"

typedef tuple<int, int, int, int> BOX;

class Controller {
  static auto addFeature(Mat& features, Mat s) -> void;
  static auto listdir(const char* path) -> vector<string>;
  static auto pascal(char* input) -> vector<BOX>;
  public:
  static void show_usage();
  static void extract(Descriptor* desc, char* dir, char* output);
  static void train(char* pos, char* neg, char* output);
  static void predict(Descriptor* desc, char* set, char* model);
  static void generate(char* input, char* output, int width, int height, int h_stride, int v_stride);
  static void detect(char* input, char* annotations);
};

#endif
