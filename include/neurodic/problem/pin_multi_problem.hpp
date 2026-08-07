/**
 * Placeholder contract for the pairwise multi-camera PIN-DIC route.
 *
 * This route is deliberately separate from NDeF: every selected camera pair
 * owns its reference-pair ROI and its three planar PIN registrations.  The
 * pair products are reconstructed independently before a later fusion stage.
 */
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "neurodic/calibration/camera_model.hpp"
#include "neurodic/problem/pin_problem.hpp"

namespace neurodic {

struct PINMultiPairProblem {
    std::string pair_id;  // Stable label, for example "cam_00__cam_01".
    PINProblem reference_stereo;  // A(t0) -> B(t0)
    PINProblem left_temporal;     // A(t0) -> A(tk)
    PINProblem deformed_stereo;   // A(t0) -> B(tk)
    CameraModel left_camera;
    CameraModel right_camera;
};

struct PINMultiProblem {
    // Route identifier is intentionally spelled pin_multi_slover for config
    // compatibility with the requested route name.
    std::string route_id{"pin_multi_slover"};
    std::vector<PINMultiPairProblem> pairs;
    double world_scale{1.0};
    bool remove_rigid_body_motion{false};

    void validate() const;
};

}  // namespace neurodic
