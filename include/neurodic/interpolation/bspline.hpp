/**
 * B-spline degree validation.
 *
 * Responsibilities: define the single interpolation family supported by NeuroDIC.
 * Inputs: requested degree.
 * Outputs: validation result or exception.
 * Ownership: value-only helpers.
 * Differentiable: NO. Degree validation is configuration logic.
 * TODO(NeuroDIC): keep this as the only interpolation-family gate.
 */
#pragma once

namespace neurodic {

bool is_supported_bspline_degree(int degree);
void validate_bspline_degree(int degree);

}  // namespace neurodic
