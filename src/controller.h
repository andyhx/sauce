#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "global.h"
#include "descriptor.h"

class Controller {
  static auto add_feature(Mat& features, Mat s) -> void;
  static auto listdir(const char* path) -> vector<string>;
  static auto pascal(string input, float scale) -> vector<BOX>;
  static auto extract_features(Descriptor* desc, Mat image) -> Mat; 
  static auto is_false_positive(BOX detection, vector<BOX>& boxes) -> bool;
  static auto count_true_positives(vector<BOX>& detections, vector<BOX>& boxes) -> int;
  static auto annotation_file(char* annotations, string file) -> string;

  public:
  static void show_usage();
  static void extract(Descriptor* desc, char* dir, char* output, float probability = 1.0);
  static void train(char* pos, char* neg, char* output);
  static void join_sets(char* a, char* b, char* output);
  static RESULT predict(Descriptor* desc, char* set, char* model);
  static void detect(Descriptor* desc, char* model, char* input, char* annotations);
  static void false_positives(Descriptor* desc, char* model, char* input, char* output);

  static void generate(char* input, char* output, int width, int height, int h_stride, int v_stride, int nrand);
};

#endif
