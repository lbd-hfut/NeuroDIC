/**
 * Result filtering.
 *
 * Responsibilities: remove invalid/outlier values after solving.
 * Inputs: result tensors and masks.
 * Outputs: filtered tensors.
 * Ownership: tensor references only.
 * Differentiable: NO for exported analysis.
 * TODO(NeuroDIC): define robust filtering policy.
 */
#pragma once

#include <torch/torch.h>

namespace neurodic {

struct SurfaceCleanupResult {
    torch::Tensor inlier_mask;       // CPU bool [N]
    torch::Tensor neighbor_distance; // CPU float64 [N], median k-NN distance
    torch::Tensor plane_residual;    // CPU float64 [N], point-to-local-plane distance
    double neighbor_distance_median{0.0};
    double neighbor_distance_mad{0.0};
    double neighbor_distance_threshold{0.0};
    double plane_residual_median{0.0};
    double plane_residual_mad{0.0};
    double plane_residual_threshold{0.0};
};

// Robust cleanup for a voxel-deduplicated multi-camera surface.  It uses an
// in-core KD-tree for k-NN queries (not an O(N^2) distance matrix), then
// rejects sparse points and points that do not lie on their local PCA plane.
SurfaceCleanupResult clean_pin_multi_surface(const torch::Tensor& points,
                                             int64_t k_neighbors = 16,
                                             double mad_factor = 5.0);

// CPU KD-tree k-NN indices for finite [N,3] points.  The queried point itself
// is excluded, making the result directly suitable for local field gradients.
torch::Tensor knn_indices_3d(const torch::Tensor& points, int64_t k_neighbors);

struct MeshCleanupResult {
    torch::Tensor face_mask;     // CPU bool [M]
    torch::Tensor face_quality;  // CPU float64 [M], 0 for rejected degenerates
    double mean_edge_length{0.0};
    double overlap_distance{0.0};
};

// MultiDIC-style mesh cleanup for stitched surfaces: reject degenerate and
// duplicate faces, then resolve near-overlapping non-adjacent faces by keeping
// the better-shaped triangle. `overlap_distance <= 0` uses 0.2 mean edge
// because centroid proximity is a conservative proxy for MultiDIC ray overlap.
MeshCleanupResult clean_pin_multi_mesh(const torch::Tensor& vertices, const torch::Tensor& faces,
                                       const torch::Tensor& quality = torch::Tensor(),
                                       double overlap_distance = 0.0,
                                       double min_triangle_quality = 0.20);

struct LocalDisplacementConsistencyResult {
    torch::Tensor predicted_displacement; // CPU float64 [N,3]
    torch::Tensor residual;               // CPU float64 [N], ||u - u_local||
    torch::Tensor inlier_mask;            // CPU bool [N]
    double residual_median{0.0};
    double residual_mad{0.0};
    double residual_threshold{0.0};
};

// Leave-one-out local-affine consistency of a reconstructed 3D displacement
// field.  It provides a closure-quality proxy independent of any point's own
// triangulation: its motion must agree with the motion implied by neighbours.
LocalDisplacementConsistencyResult compute_local_displacement_consistency(
    const torch::Tensor& coordinates, const torch::Tensor& displacement,
    const torch::Tensor& valid = torch::Tensor(), int64_t k_neighbors = 16,
    double mad_factor = 5.0);

}  // namespace neurodic
