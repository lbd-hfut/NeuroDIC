"""Read-only NDeF workflow contract; surface and precalculation are DAG siblings."""

CANONICAL_SOLVER = "ndef"
CASE_KEY = "ndef_multi"


def stages():
    return [
        ("ndef.inputs", [], ["images", "calibration"], [], "preparation"),
        ("ndef.roi", ["ndef.inputs"], [], ["roi_masks"], "separate_python_call"),
        ("ndef.surface", ["ndef.inputs", "ndef.roi"], [], ["reference_surface"], "combined_surface_call"),
        ("ndef.precalculation", ["ndef.inputs", "ndef.roi", "ndef.surface"], [], ["sparse_tracks", "sparse_scale"], "separate_python_call"),
        ("ndef.deformation.train", ["ndef.inputs", "ndef.roi", "ndef.surface", "ndef.precalculation"], [], ["checkpoint", "training_history"], "deformation_combined_call"),
        ("ndef.deformation.infer", ["ndef.deformation.train"], [], ["deformation"], "deformation_combined_call"),
        ("ndef.postprocess", ["ndef.deformation.infer"], [], ["deformation"], "deformation_combined_call"),
        ("ndef.evaluate", ["ndef.postprocess"], [], ["evaluation"], "deformation_combined_call"),
    ]
