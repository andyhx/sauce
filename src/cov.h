#ifndef COV_H
#define COV_H
#include "global.h"
#include "descriptor.h"

class Cov : public Descriptor {
  static auto convert_to_grayscale(Acc a) -> Acc;
  static auto calculate_covariances(Acc a) -> Acc;
  static auto normalize_blocks(Acc a) -> Acc;
  static auto flatten_features(Acc a) -> Acc;

  public:
  virtual vector<function<Acc(Acc)>> features(); 
};
#endif
