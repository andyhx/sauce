#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H
#include "global.h"

class Descriptor {
  public:
  Acc iterate(Acc a);
  virtual vector<function<Acc(Acc)>> features() = 0;
};

#endif
