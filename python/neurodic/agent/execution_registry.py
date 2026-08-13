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
    "pin_multi.separate_pair_roi_call": GuardedActionCapability(
        "pin_multi.separate_pair_roi_call", True, "scope.pair_id", "requested_action_only",
        "CPU-only guarded single-pair pair_roi adapter; it does not complete the full PIN Multi trial."),
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
