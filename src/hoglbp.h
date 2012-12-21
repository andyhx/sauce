#ifndef HOGLBP_H
#define HOGLBP_H
#include "global.h"
#include "descriptor.h"
#include "hog.h"
#include "lbp.h"

class HOGLBP : public Descriptor {
  Acc iterate(Acc acc);
  vector<function<Acc(Acc)>> features();

  static int hogCell;
  static int lbpBlock;
};
#endif
