#ifndef EDGE_H
#define EDGE_H
#include "global.h"
#include "descriptor.h"

class Edge : public Descriptor {
  static auto convert_to_grayscale(Acc a) -> Acc;
  static auto calculate_gradient(Acc a) -> Acc;
  static auto calculate_magnitude_orientation(Acc a) -> Acc;
  static auto calculate_affinities(Acc a) -> Acc;

  public:
  virtual vector<function<Acc(Acc)>> features(); 

  static Mat edgelets;
  static int n;
};
#endif
