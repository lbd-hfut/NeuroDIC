"""Single native-free truth source for guarded execution capability."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

@dataclass(frozen=True)
class GuardedActionCapability:
    action_id: str
    execution_supported: bool
    scope_requirement: str | None
    completion_scope: str
    notes: str
    def to_dict(self) -> dict[str, Any]:
        return {"execution_supported": self.execution_supported, "scope_requirement": self.scope_requirement,
                "completion_scope": self.completion_scope, "capability_notes": self.notes}

# Adapter registry only; conceptual stages/actions remain owned by Loop 2/6.
GUARDED_ACTIONS: Mapping[str, GuardedActionCapability] = {
    "pin_stereo.combined_solver_call": GuardedActionCapability(
        "pin_stereo.combined_solver_call", True, "scope.selected_frame", "combined_action",
        "Guarded single-frame Stereo PIN call. It atomically covers all three field solves, reconstruction, postprocess, and evaluation; fields cannot be selected or reused independently."),
    "pin.combined_solver_call": GuardedActionCapability(
        "pin.combined_solver_call", True, "scope.selected_frame", "combined_action",
        "Guarded single-frame planar PIN call. It atomically covers initialization, training, inference, postprocess, and evaluation; conceptual stages cannot be selected independently."),
    "pin_multi.separate_pair_roi_call": GuardedActionCapability(
        "pin_multi.separate_pair_roi_call", True, "scope.pair_id", "requested_action_only",
        "CPU-only guarded single-pair pair_roi adapter; it does not complete the full PIN Multi trial."),
    "pin_multi.pair_solve_quality_call": GuardedActionCapability(
        "pin_multi.pair_solve_quality_call", True, "scope.pair_id + scope.selected_frame", "combined_action",
        "Guarded one-pair PIN Multi solve plus its native quality output; it is pair-local partial execution, never a full PIN Multi result."),
    "pin_multi.fusion_postprocess_call": GuardedActionCapability(
        "pin_multi.fusion_postprocess_call", True, "scope.selected_frame + scope.planned_pair_ids", "combined_action",
        "Guarded managed-only PIN Multi fusion, surface cleanup, strain, and conditional mesh. It excludes evaluation and accepts no legacy discovery."),
    "ndef.combined_surface_call": GuardedActionCapability(
        "ndef.combined_surface_call", True, "managed NDeF inputs + managed ROI + manual dense batch", "combined_action",
        "Guarded NDeF reference-surface action. It atomically covers sparse training, same-model dense continuation, full-field inference, fusion, and export; no partial surface stage is reusable."),
    "ndef.roi.generate_call": GuardedActionCapability(
        "ndef.roi.generate_call", True, "managed calibration package + ordered references + ROI options", "combined_action",
        "Guarded atomic NDeF ROI generation. It publishes only validated per-camera masks, bundle, and metadata; it does not execute surface training."),
    "ndef.precalculation_call": GuardedActionCapability(
        "ndef.precalculation_call", True, "managed calibration + surface + ordered ROI + explicit sparse options", "requested_action_only",
        "Guarded managed NDeF sparse-precalculation call. It publishes only sparse tracks and scale metadata; resume is unsupported and no deformation action is implied."),
    "ndef.deformation_combined_call": GuardedActionCapability(
        "ndef.deformation_combined_call", True, "managed D/E + calibration/images/frame + explicit model/training + manual batch + fresh init", "combined_action",
        "Guarded managed NDeF deformation train/infer/postprocess action with optional internal fixed evaluation; no split train/infer action or checkpoint resume."),
}

def capability_for(action_id: str) -> GuardedActionCapability:
    return GUARDED_ACTIONS.get(action_id, GuardedActionCapability(
        action_id, False, None, "not_executable", "No verified guarded execution adapter is registered for this conceptual action."))

def annotate_actions(actions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{**dict(action), **capability_for(str(action["action_id"])).to_dict()} for action in actions]

def capability_summary(actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    annotated = annotate_actions(actions); supported = [item["action_id"] for item in annotated if item["execution_supported"]]
    return {"execution_supported": bool(supported), "supported_action_ids": supported,
            "partial_execution_possible": any(item["completion_scope"] == "requested_action_only" for item in annotated)}
