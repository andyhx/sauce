#include "acc.h"

class HOG {
  static auto convert_to_grayscale(Acc& a) -> Acc&;
  static auto calculate_gradient(Acc& a) -> Acc&;
  static auto calculate_magnitude_orientation(Acc& a) -> Acc&;
  static auto perform_binning(Acc& a) -> Acc&;
  static auto normalize_blocks(Acc& a) -> Acc&;
  
  public:
  static vector<function<Acc&(Acc&)>> steps();
};
