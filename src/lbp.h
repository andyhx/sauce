#ifndef LBP_H
#define LBP_H
#include "global.h"
#include "descriptor.h"

class LBP : public Descriptor {
  static auto convert_to_grayscale(Acc a) -> Acc;
  static auto threshold_masks(Acc a) -> Acc;
  static auto flatten_features(Acc a) -> Acc;

  public:
  virtual vector<function<Acc(Acc)>> features(); 

  static int blockWidth; 
  static int threshold;
};
#endif
