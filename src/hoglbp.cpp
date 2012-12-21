#include "hoglbp.h"

int HOGLBP::hogCell = 8;
int HOGLBP::lbpBlock = 32;

Acc HOGLBP::iterate(Acc acc) {
  HOG* hog = new HOG();
  HOG::cellWidth = hogCell;
  LBP* lbp = new LBP();
  LBP::blockWidth = lbpBlock; 
  Acc acc1 = hog->iterate(acc);
  Acc acc2 = lbp->iterate(acc);  

  Mat m1 = acc1.m.t();
  Mat m2 = acc2.m.t();
  m1.push_back(m2);
  acc.m = m1.t();

  return acc;
}

vector<function<Acc(Acc)>> HOGLBP::features() {
    vector<function<Acc(Acc)>> funs;
    return funs; 
};
