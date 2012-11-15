#ifndef HOG_H
#define HOG_H
#include "global.h"

class HOG {
  static auto convert_to_grayscale(Acc a) -> Acc;
  static auto calculate_gradient(Acc a) -> Acc;
  static auto calculate_magnitude_orientation(Acc a) -> Acc;
  static auto perform_binning(Acc a) -> Acc;
  static auto normalize_blocks(Acc a) -> Acc;
  static auto flatten_features(Acc a) -> Acc;
  
  public:
  static auto iterate(vector<function<Acc(Acc)>> through, Acc a) -> Acc;
  static vector<function<Acc(Acc)>> features();
};
#endif
