#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "global.h"
#include "descriptor.h"

typedef tuple<int, int, int, int> BOX;
#define ELEMENT(N, TUPLE) get<N>(TUPLE)

class Controller {
  static auto add_feature(Mat& features, Mat s) -> void;
  static auto listdir(const char* path) -> vector<string>;
  static auto pascal(string input, float scale) -> vector<BOX>;
  static auto extract_features(Descriptor* desc, Mat image) -> Mat; 
  static auto is_false_positive(BOX detection, vector<BOX>& boxes) -> bool;

  public:
  static void show_usage();
  static void extract(Descriptor* desc, char* dir, char* output);
  static void train(char* pos, char* neg, char* output);
  static void predict(Descriptor* desc, char* set, char* model);
  static void detect(Descriptor* desc, char* model, char* input, char* annotations);

  static void generate(char* input, char* output, int width, int height, int h_stride, int v_stride);
};

#endif
