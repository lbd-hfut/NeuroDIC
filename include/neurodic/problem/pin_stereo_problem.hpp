/** Stereo PIN problem using three left-reference planar PIN fields. */
#pragma once

#include "neurodic/calibration/camera_model.hpp"
#include "neurodic/geometry/triangulation.hpp"
#include "neurodic/problem/pin_problem.hpp"

namespace neurodic {

struct PINStereoProblem {
    PINProblem reference_disparity;  // L0 -> R0
    PINProblem left_temporal;        // L0 -> L1
    PINProblem deformed_disparity;   // L0 -> R1
    CameraModel left_camera;
    CameraModel right_camera;
    ReconstructionOptions reconstruction;
    double world_scale{1.0};
    bool require_image_bounds{true};

    PINStereoProblem(PINProblem reference_disparity_problem, PINProblem left_temporal_problem,
                     PINProblem deformed_disparity_problem, CameraModel left, CameraModel right);
    void validate() const;
};

}  // namespace neurodic
