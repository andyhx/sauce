#ifndef SLBP_H
#define SLBP_H
#include "global.h"
#include "descriptor.h"

class SLBP : public Descriptor {
  static auto convert_to_grayscale(Acc a) -> Acc;
  static auto threshold_masks(Acc a) -> Acc;
  static auto perform_binning(Acc a) -> Acc;
  static auto normalize_blocks(Acc a) -> Acc;
  static auto flatten_features(Acc a) -> Acc;

  public:
  virtual vector<function<Acc(Acc)>> features(); 

  static int blockWidth; 
  static int threshold;
};
#endif
