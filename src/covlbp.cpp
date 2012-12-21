#include "covlbp.h"

int CovLBP::covBlock = 16;
int CovLBP::lbpBlock = 32;

Acc CovLBP::iterate(Acc acc) {
  Cov* cov = new Cov();
  Cov::blockWidth = covBlock;
  LBP* lbp = new LBP();
  LBP::blockWidth = lbpBlock; 
  Acc acc1 = cov->iterate(acc);
  Acc acc2 = lbp->iterate(acc);  

  Mat m1 = acc1.m.t();
  Mat m2 = acc2.m.t();
  m1.push_back(m2);
  acc.m = m1.t();

  return acc;
}

vector<function<Acc(Acc)>> CovLBP::features() {
    vector<function<Acc(Acc)>> funs;
    return funs; 
};
