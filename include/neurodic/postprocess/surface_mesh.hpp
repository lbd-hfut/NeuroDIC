/** Local tangent-plane triangle mesh reconstruction for cleaned DIC surfaces. */
#pragma once

#include <torch/torch.h>

namespace neurodic {

struct SurfaceMeshOptions {
    int64_t k_neighbors{16};
    // <= 0 selects 2.5 times the median nearest-neighbour spacing.
    double max_edge_length{0.0};
    // Equilateral=1; sliver/degenerate=0.
    double min_triangle_quality{0.20};
};

struct SurfaceMesh {
    torch::Tensor vertices;  // CPU float64 [N,3]
    torch::Tensor faces;     // CPU int64 [M,3]
    torch::Tensor normals;   // CPU float64 [N,3]
    torch::Tensor quality;   // CPU float64 [M]
    double median_spacing{0.0};
    double max_edge_length{0.0};
};

// Reconstruct a local surface mesh.  This intentionally is not global 3D
// Delaunay, which would join opposite sides of a cylinder or bridge holes.
SurfaceMesh triangulate_pin_multi_surface(const torch::Tensor& points,
                                          SurfaceMeshOptions options = {});

}  // namespace neurodic
