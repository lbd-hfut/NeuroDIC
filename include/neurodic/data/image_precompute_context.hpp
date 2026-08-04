/** Shared image padding and B-spline precomputation for seed and neural paths. */
#pragma once

#include <torch/torch.h>

#include "neurodic/interpolation/bspline_coefficients.hpp"

namespace neurodic {

struct ImagePrecomputeOptions {
    int integer_search_radius{0};
    int coarse_subset_radius{0};
    int fine_subset_radius{0};
    int subset_radius{0};
    int bspline_border{3};
    int bspline_degree{5};
};

int calculate_image_padding(const ImagePrecomputeOptions& options);
torch::Tensor mirror_pad_image(const torch::Tensor& image, int pad);
torch::Tensor zero_pad_roi_mask(const torch::Tensor& mask, int pad);

class ImagePrecomputeContext {
public:
    static ImagePrecomputeContext create(
        const torch::Tensor& reference_image,
        const torch::Tensor& deformed_image,
        const torch::Tensor& roi_mask,
        const ImagePrecomputeOptions& options
    );

    torch::Tensor original_to_padded(const torch::Tensor& xy) const;
    torch::Tensor padded_to_original(const torch::Tensor& xy) const;

    torch::Tensor reference_padded;
    torch::Tensor deformed_padded;
    torch::Tensor roi_mask_padded;
    int pad_offset{0};
    BSplineCoefficientBlock reference_coefficients;
    BSplineCoefficientBlock deformed_coefficients;
};

}  // namespace neurodic
