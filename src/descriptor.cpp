#include "descriptor.h"

Acc Descriptor::iterate(Acc acc) {
  for(function<Acc(Acc)>& fun : this->features()) {
    acc = fun(acc);
  }
  return acc;
}

