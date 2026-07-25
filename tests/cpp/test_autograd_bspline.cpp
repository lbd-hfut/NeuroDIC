#include "neurodic/interpolation/torch_bspline.hpp"

void test_autograd_bspline() {
    // TODO(NeuroDIC):
    // 1. loss = sampler(coefficients, coordinates).sum()
    // 2. loss.backward()
    // 3. coordinates.grad() must exist
    // 4. compare against finite differences
    // 5. add torch::autograd::gradcheck where applicable
    // 6. add CPU/GPU consistency checks
}
