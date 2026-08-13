/**
 * Prepared scalar fields on triangular surfaces for visualization.
 *
 * This is intentionally a CPU postprocess step: it validates every triangle
 * once and reduces vertex fields to face fields before Python renders a
 * bounded display subset.
 */
#pragma once

#include <torch/torch.h>

namespace neurodic {

struct SurfaceFaceField {
    torch::Tensor face_centers;  // CPU float64 [M, 3], NaN for invalid faces.
    torch::Tensor face_values;   // CPU float64 [M, C], NaN for invalid faces.
    torch::Tensor valid_faces;   // CPU bool [M].
};

// `points` is [N,3], `faces` is [M,3], and `point_values` is [N,C].  A face
// is valid only if its indices, coordinates, and every scalar component are
// finite.  Face values are the arithmetic mean of the three vertex values.
SurfaceFaceField prepare_surface_face_field(const torch::Tensor& points,
                                            const torch::Tensor& faces,
                                            const torch::Tensor& point_values);

}  // namespace neurodic
