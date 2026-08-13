"""Read-only stereo PIN workflow contract."""

CANONICAL_SOLVER = "pin_stereo"
CASE_KEY = "pin_stereo"


def stages():
    return [
        ("stereo.inputs", [], ["left_images", "right_images", "roi", "camera_pair"], [], "preparation"),
        ("stereo.planar_fields", ["stereo.inputs"], [], ["reference_disparity", "left_temporal", "deformed_disparity"], "combined_solver_call"),
        ("stereo.reconstruct", ["stereo.planar_fields"], [], ["reference_reconstruction", "current_reconstruction", "deformation"], "combined_solver_call"),
        ("stereo.postprocess", ["stereo.reconstruct"], [], ["deformation"], "combined_solver_call"),
        ("stereo.evaluate", ["stereo.postprocess"], [], ["evaluation"], "combined_solver_call"),
    ]
