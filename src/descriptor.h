#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H
#include "global.h"

class Descriptor {
  public:
  virtual Acc iterate(Acc acc) {
    for(function<Acc(Acc)>& fun : this->features()) {
      acc = fun(acc);
    }
    return acc;
  };
  virtual vector<function<Acc(Acc)>> features() = 0;
};

#endif
