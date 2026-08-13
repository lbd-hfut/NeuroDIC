#include "neurodic/postprocess/surface_field.hpp"

#include <ATen/Parallel.h>

#include <cmath>
#include <limits>

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

SurfaceFaceField prepare_surface_face_field(const torch::Tensor& points_input,
                                            const torch::Tensor& faces_input,
                                            const torch::Tensor& values_input) {
    if (!points_input.defined() || points_input.dim() != 2 || points_input.size(1) != 3 ||
        !faces_input.defined() || faces_input.dim() != 2 || faces_input.size(1) != 3 ||
        !values_input.defined() || values_input.dim() != 2 || values_input.size(0) != points_input.size(0) ||
        values_input.size(1) < 1)
        throw ValidationError("Surface face field expects points [N,3], faces [M,3], and point_values [N,C]");

    auto points = points_input.detach().to(torch::kCPU).to(torch::kFloat64).contiguous();
    auto faces = faces_input.detach().to(torch::kCPU).to(torch::kInt64).contiguous();
    auto values = values_input.detach().to(torch::kCPU).to(torch::kFloat64).contiguous();
    const auto face_count = faces.size(0), components = values.size(1), point_count = points.size(0);
    const auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64);
    auto centers = torch::full({face_count, 3}, std::numeric_limits<double>::quiet_NaN(), options);
    auto face_values = torch::full({face_count, components}, std::numeric_limits<double>::quiet_NaN(), options);
    auto valid = torch::zeros({face_count}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kBool));
    const auto point = points.accessor<double, 2>();
    const auto face = faces.accessor<int64_t, 2>();
    const auto value = values.accessor<double, 2>();
    auto center = centers.accessor<double, 2>();
    auto field = face_values.accessor<double, 2>();
    auto usable = valid.accessor<bool, 1>();

    at::parallel_for(0, face_count, 8192, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            const auto a = face[row][0], b = face[row][1], c = face[row][2];
            if (a < 0 || b < 0 || c < 0 || a >= point_count || b >= point_count || c >= point_count) continue;
            const int64_t ids[3] = {a, b, c};
            bool finite = true;
            for (const auto id : ids) {
                for (int dimension = 0; dimension < 3; ++dimension)
                    finite = finite && std::isfinite(point[id][dimension]);
                for (int64_t component = 0; component < components; ++component)
                    finite = finite && std::isfinite(value[id][component]);
            }
            if (!finite) continue;
            for (int dimension = 0; dimension < 3; ++dimension)
                center[row][dimension] = (point[a][dimension] + point[b][dimension] + point[c][dimension]) / 3.0;
            for (int64_t component = 0; component < components; ++component)
                field[row][component] = (value[a][component] + value[b][component] + value[c][component]) / 3.0;
            usable[row] = true;
        }
    });
    return {centers, face_values, valid};
}

}  // namespace neurodic
