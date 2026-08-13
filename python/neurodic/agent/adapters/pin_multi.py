"""Read-only pairwise multiview PIN workflow contract."""

CANONICAL_SOLVER = "pin_multi"
CASE_KEY = "pin_multi"


def stages():
    return [
        ("pin_multi.inputs", [], ["images", "calibration"], [], "preparation"),
        ("pin_multi.pair_select", ["pin_multi.inputs"], [], ["pair_selection"], "separate_pair_roi_call"),
        ("pin_multi.pair_roi", ["pin_multi.pair_select"], [], ["pair_roi"], "separate_pair_roi_call"),
        ("pin_multi.pair_solve", ["pin_multi.pair_roi"], [], ["pair_products"], "combined_solver_call"),
        ("pin_multi.pair_quality", ["pin_multi.pair_solve"], [], ["pair_quality"], "combined_solver_call"),
        ("pin_multi.fusion", ["pin_multi.pair_quality"], [], ["fused_surface"], "combined_solver_call"),
        ("pin_multi.postprocess", ["pin_multi.fusion"], [], ["fused_surface"], "combined_solver_call"),
        ("pin_multi.evaluate", ["pin_multi.postprocess"], [], ["evaluation"], "combined_solver_call"),
    ]
