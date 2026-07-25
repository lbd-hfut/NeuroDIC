/**
 * Single ROI representation.
 *
 * Responsibilities: define one continuous region solved by one neural field.
 * Inputs: rectangular bounds for the initial skeleton.
 * Outputs: contains checks and validation.
 * Ownership: value object.
 * Differentiable: NO. ROI membership and sampling preparation are preprocessing.
 * TODO(NeuroDIC):
 * 1. Add mask/polygon representations.
 * 2. Convert ROI to sampling coordinates.
 * 3. Preserve the one ROI -> one field invariant.
 */
#pragma once

namespace neurodic {

class ROI {
public:
    ROI() = default;
    ROI(double x_min, double y_min, double x_max, double y_max);

    [[nodiscard]] bool contains(double x, double y) const;
    void validate() const;

    [[nodiscard]] double x_min() const noexcept { return x_min_; }
    [[nodiscard]] double y_min() const noexcept { return y_min_; }
    [[nodiscard]] double x_max() const noexcept { return x_max_; }
    [[nodiscard]] double y_max() const noexcept { return y_max_; }

private:
    double x_min_ = 0.0;
    double y_min_ = 0.0;
    double x_max_ = 1.0;
    double y_max_ = 1.0;
};

}  // namespace neurodic
