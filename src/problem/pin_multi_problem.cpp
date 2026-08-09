#include "neurodic/problem/pin_multi_problem.hpp"

#include <cmath>

#include "neurodic/core/exceptions.hpp"

namespace neurodic {

void PINMultiProblem::validate() const {
    if (route_id != "pin_multi_slover")
        throw ValidationError("PIN multi route_id must be pin_multi_slover");
    if (pairs.empty())
        throw ValidationError("PIN multi problem requires at least one camera pair");
    if (!std::isfinite(world_scale) || world_scale <= 0.0)
        throw ValidationError("PIN multi world_scale must be finite and positive");
    for (const auto& pair : pairs) {
        if (pair.pair_id.empty())
            throw ValidationError("PIN multi pair_id must be non-empty");
        for (const auto& other : pairs) {
            if (&pair != &other && pair.pair_id == other.pair_id)
                throw ValidationError("PIN multi pair_id must be unique");
        }
        pair.reference_stereo.validate();
        pair.left_temporal.validate();
        pair.deformed_stereo.validate();
        pair.left_camera.validate();
        pair.right_camera.validate();
        const auto shape = pair.reference_stereo.reference_image.sizes();
        if (pair.left_temporal.reference_image.sizes() != shape ||
            pair.deformed_stereo.reference_image.sizes() != shape ||
            pair.reference_stereo.roi_mask.sizes() != pair.left_temporal.roi_mask.sizes() ||
            pair.reference_stereo.roi_mask.sizes() != pair.deformed_stereo.roi_mask.sizes() ||
            !torch::equal(pair.reference_stereo.roi_mask, pair.left_temporal.roi_mask) ||
            !torch::equal(pair.reference_stereo.roi_mask, pair.deformed_stereo.roi_mask))
            throw ValidationError("PIN multi pair fields must share the L0 image shape and ROI");
    }
}

}  // namespace neurodic
