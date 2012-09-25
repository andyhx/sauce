#include "acc.h"

class HOG {
  static auto convert_to_grayscale(Acc& a) -> Acc&;
  static auto calculate_gradient(Acc& a) -> Acc&;
  
  public:
  static vector<function<Acc&(Acc&)>> steps();
};
