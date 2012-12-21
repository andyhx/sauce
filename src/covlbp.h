#ifndef COVLBP_H
#define COVLBP_H
#include "global.h"
#include "descriptor.h"
#include "cov.h"
#include "lbp.h"

class CovLBP : public Descriptor {
  Acc iterate(Acc acc);
  vector<function<Acc(Acc)>> features();

  static int covBlock;
  static int lbpBlock;
};
#endif
