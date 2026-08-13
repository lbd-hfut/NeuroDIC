"""Read-only PIN 2D workflow contract."""

CANONICAL_SOLVER = "pin"
CASE_KEY = "pin_2d"


def stages():
    return [
        ("pin.inputs", [], ["reference_image", "roi_image"], [], "preparation"),
        ("pin.initialization", ["pin.inputs"], [], ["seed_artifact"], "combined_solver_call"),
        ("pin.train", ["pin.initialization"], [], ["pin_result"], "combined_solver_call"),
        ("pin.infer", ["pin.train"], [], ["pin_result"], "combined_solver_call"),
        ("pin.postprocess", ["pin.infer"], [], ["pin_result"], "combined_solver_call"),
        ("pin.evaluate", ["pin.postprocess"], [], ["evaluation"], "combined_solver_call"),
    ]
