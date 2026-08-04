#include "neurodic/interpolation/bspline_coefficients.hpp"

#include <cmath>
#include <vector>

#include "neurodic/core/exceptions.hpp"
#include "neurodic/interpolation/bspline.hpp"

namespace neurodic {
namespace {

double factorial(int n) {
    double value = 1.0;
    for (int i = 2; i <= n; ++i) value *= i;
    return value;
}

double combination(int n, int k) {
    return factorial(n) / (factorial(k) * factorial(n - k));
}

double basis(double x, int derivative, int degree) {
    const double factor = factorial(degree) / factorial(degree - derivative);
    double sum = 0.0;
    for (int k = 0; k <= degree + 1; ++k) {
        const double shifted = x + (degree + 1) / 2.0 - k;
        if (shifted > 0.0) {
            sum += (k % 2 ? -1.0 : 1.0) * combination(degree + 1, k) *
                   factor * std::pow(shifted, degree - derivative);
        }
    }
    return sum / factorial(degree);
}

torch::Tensor qk_matrix(int degree, const torch::TensorOptions& options) {
    const int n = degree + 1;
    const int offset = degree / 2;
    std::vector<double> values(static_cast<std::size_t>(n * n));
    for (int derivative = 0; derivative < n; ++derivative) {
        for (int col = 0; col < n; ++col) {
            values[static_cast<std::size_t>(derivative * n + col)] =
                (derivative % 2 ? -1.0 : 1.0) * basis(-offset + col, derivative, degree) /
                factorial(derivative);
        }
    }
    return torch::from_blob(values.data(), {n, n}, torch::TensorOptions().dtype(torch::kFloat64))
        .clone().to(options);
}

torch::Tensor prefilter_axis(const torch::Tensor& input, int64_t axis, int degree) {
    const int64_t length = input.size(axis);
    const int radius = degree / 2;
    auto kernel = torch::zeros({length}, input.options());
    kernel.index_put_({0}, basis(0.0, 0, degree));
    for (int i = 1; i <= radius; ++i) {
        const double value = basis(i, 0, degree);
        kernel.index_put_({i}, value);
        kernel.index_put_({length - i}, value);
    }
    auto spectrum = torch::fft::fft(input, c10::nullopt, axis);
    auto kernel_spectrum = torch::fft::fft(kernel).reshape(
        axis == 0 ? std::vector<int64_t>{length, 1} : std::vector<int64_t>{1, length});
    return torch::real(torch::fft::ifft(spectrum / kernel_spectrum, c10::nullopt, axis));
}

}  // namespace

void BSplineCoefficientBlock::validate() const {
    validate_bspline_degree(degree);
    if (height <= 0 || width <= 0 || pad_offset < 0 || !coeff_cpu.defined() ||
        coeff_cpu.device().is_cuda() || coeff_cpu.dim() != 4 ||
        coeff_cpu.size(0) != height || coeff_cpu.size(1) != width ||
        coeff_cpu.size(2) != degree + 1 || coeff_cpu.size(3) != degree + 1) {
        throw ValidationError("Invalid B-spline coefficient block");
    }
}

const torch::Tensor& BSplineCoefficientBlock::cpu() const {
    validate();
    return coeff_cpu;
}

const torch::Tensor& BSplineCoefficientBlock::on(const torch::Device& device) const {
    validate();
    if (device.is_cpu()) return coeff_cpu;
    if (!coeff_gpu.defined() || coeff_gpu.device() != device) coeff_gpu = coeff_cpu.to(device);
    return coeff_gpu;
}

torch::Tensor compute_bspline_coefficients(const torch::Tensor& image, int degree) {
    validate_bspline_degree(degree);
    if (!image.defined() || image.dim() != 2 || !image.is_floating_point()) {
        throw ValidationError("B-spline image must be a floating [H,W] tensor");
    }
    if (!image.device().is_cpu()) {
        throw ValidationError("B-spline preprocessing must run once on CPU");
    }
    if (image.size(0) < degree + 1 || image.size(1) < degree + 1) {
        throw ValidationError("B-spline image dimensions must be at least degree+1");
    }
    torch::NoGradGuard guard;
    auto coefficients = image.contiguous();
    if (degree > 1) {
        coefficients = prefilter_axis(coefficients, 1, degree);
        coefficients = prefilter_axis(coefficients, 0, degree);
    }

    const int n = degree + 1;
    const int offset = degree / 2;
    auto ys = torch::arange(image.size(0), torch::TensorOptions().dtype(torch::kLong));
    auto xs = torch::arange(image.size(1), torch::TensorOptions().dtype(torch::kLong));
    auto taps = torch::arange(n, torch::TensorOptions().dtype(torch::kLong));
    auto yi = (ys.unsqueeze(1) + taps - offset).clamp(0, image.size(0) - 1);
    auto xi = (xs.unsqueeze(1) + taps - offset).clamp(0, image.size(1) - 1);
    auto blocks = coefficients.index({yi.unsqueeze(1).unsqueeze(3), xi.unsqueeze(0).unsqueeze(2)});
    auto qk = qk_matrix(degree, image.options());
    return torch::einsum("ij,hwjk,kl->hwil", {qk, blocks, qk.transpose(0, 1)}).contiguous();
}

BSplineCoefficientBlock make_bspline_coefficient_block(
    const torch::Tensor& mirror_padded_image, int degree, int pad_offset) {
    if (pad_offset < 0) throw ValidationError("pad_offset must be non-negative");
    auto cpu_image = mirror_padded_image.detach().to(torch::kCPU).contiguous();
    BSplineCoefficientBlock result;
    result.height = static_cast<int>(cpu_image.size(0));
    result.width = static_cast<int>(cpu_image.size(1));
    result.degree = degree;
    result.pad_offset = pad_offset;
    result.coeff_cpu = compute_bspline_coefficients(cpu_image, degree);
    result.validate();
    return result;
}

}  // namespace neurodic
