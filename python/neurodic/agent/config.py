"""Pure, strict configuration operations for trial dry-run planning."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .schemas import canonical_json


@dataclass(frozen=True)
class ConfigChangeRecord:
    path: str
    old_value: Any
    new_value: Any
    change_kind: str = "modified"

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "old_value": self.old_value, "new_value": self.new_value,
                "change_kind": self.change_kind}


@dataclass(frozen=True)
class PolicyViolationRecord:
    path: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class OwnershipRule:
    pattern: str
    stages: Sequence[str]
    classification: str = "trial_tunable"
    notes: str = ""


# This registry deliberately owns only scientific paths.  Case identity and
# output paths are handled by protected-path policy below.
OWNERSHIP: dict[str, tuple[OwnershipRule, ...]] = {
    "pin": (
        OwnershipRule("runtime.*", ("pin.initialization",)), OwnershipRule("interpolation.*", ("pin.initialization",)),
        OwnershipRule("model.*", ("pin.train",)), OwnershipRule("initialization.*", ("pin.initialization",)),
        OwnershipRule("training.*", ("pin.train",)), OwnershipRule("evaluation.*", ("pin.evaluate",)),
    ),
    "pin_stereo": (
        OwnershipRule("runtime.*", ("stereo.planar_fields",)), OwnershipRule("interpolation.*", ("stereo.planar_fields",)),
        OwnershipRule("model.*", ("stereo.planar_fields",)), OwnershipRule("initialization.*", ("stereo.planar_fields",)),
        OwnershipRule("training.*", ("stereo.planar_fields",)), OwnershipRule("reconstruction.*", ("stereo.reconstruct",)),
        OwnershipRule("traditional_strain.*", ("stereo.postprocess",)), OwnershipRule("evaluation.*", ("stereo.evaluate",)),
    ),
    "pin_multi": (
        OwnershipRule("runtime.*", ("pin_multi.pair_select",)), OwnershipRule("camera_pairs.*", ("pin_multi.pair_select",)),
        OwnershipRule("pair_roi.generator", ("pin_multi.pair_roi",)), OwnershipRule("pair_roi.feature_method", ("pin_multi.pair_roi",)),
        OwnershipRule("pair_roi.max_features", ("pin_multi.pair_roi",)), OwnershipRule("pair_roi.match_ratio", ("pin_multi.pair_roi",)),
        OwnershipRule("pair_roi.mutual_check", ("pin_multi.pair_roi",)), OwnershipRule("pair_roi.ransac_reprojection_threshold_px", ("pin_multi.pair_roi",)),
        OwnershipRule("pair_roi.min_matches", ("pin_multi.pair_roi",)), OwnershipRule("pair_roi.support", ("pin_multi.pair_roi",)),
        OwnershipRule("pair_roi.alpha_radius_scale", ("pin_multi.pair_roi",)), OwnershipRule("pair_roi.erode_pixels", ("pin_multi.pair_roi",)),
        OwnershipRule("reconstruction.*", ("pin_multi.pair_solve",)), OwnershipRule("fusion.*", ("pin_multi.fusion",)),
        OwnershipRule("traditional_strain.*", ("pin_multi.postprocess",)), OwnershipRule("evaluation.*", ("pin_multi.evaluate",)),
    ),
    "ndef": (
        OwnershipRule("runtime.*", ("ndef.surface", "ndef.precalculation", "ndef.deformation.train")),
        OwnershipRule("surface.*", ("ndef.surface",)), OwnershipRule("surface_model.*", ("ndef.surface",)),
        OwnershipRule("surface_training.*", ("ndef.surface",)), OwnershipRule("surface_dense_training.*", ("ndef.surface",)),
        OwnershipRule("precalculation.*", ("ndef.precalculation",)),
        OwnershipRule("deformation_model.*", ("ndef.deformation.train",)), OwnershipRule("deformation_training.*", ("ndef.deformation.train",)),
        OwnershipRule("interpolation.*", ("ndef.deformation.train",)), OwnershipRule("evaluation.*", ("ndef.evaluate",)),
    ),
}


_COMMON_PROTECTED = ("case", "output", "solver", "mode", "notes")
PROTECTED: dict[str, tuple[str, ...]] = {
    "pin": _COMMON_PROTECTED + ("roi",),
    "pin_stereo": _COMMON_PROTECTED + ("roi", "reconstruction.world_scale"),
    "pin_multi": _COMMON_PROTECTED + ("pin_2d_config", "pair_roi.output", "camera_pairs", "reconstruction.world_scale"),
    "ndef": _COMMON_PROTECTED + ("scale", "precalculation.displacement"),
}

# These fields are accepted by the existing thin APIs but are deliberately
# absent from the example YAMLs because evaluation is opt-in.  Their defaults
# preserve those API semantics; this is an adapter for existing config behavior,
# not a second defaults system.
OPTIONAL_DEFAULTS: dict[str, dict[str, Any]] = {
    "pin": {"evaluation.enabled": False, "evaluation.sample_count": 0, "evaluation.seed": 0, "evaluation.patch_radius": 0},
    "pin_stereo": {"evaluation.enabled": False, "evaluation.sample_count": 0, "evaluation.seed": 0, "evaluation.patch_radius": 0},
    "pin_multi": {},
    "ndef": {"evaluation.enabled": False, "evaluation.sample_count": 0, "evaluation.seed": 0},
}


def _matches(path: str, pattern: str) -> bool:
    base = pattern[:-2] if pattern.endswith(".*") else pattern
    return path == base or (pattern.endswith(".*") and path.startswith(base + "."))


def flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Return leaf paths; empty maps are rejected by override validation."""
    result: dict[str, Any] = {}
    for key in sorted(value):
        if not isinstance(key, str) or not key:
            raise ValueError("Override keys must be non-empty strings")
        item = value[key]; path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            if not item:
                raise ValueError(f"Override cannot contain an empty mapping: {path}")
            result.update(flatten(item, path))
        else:
            result[path] = item
    return result


def lookup(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def _same_type(baseline: Any, replacement: Any) -> bool:
    if baseline is None:
        return replacement is None
    if isinstance(baseline, bool):
        return isinstance(replacement, bool)
    if isinstance(baseline, int):
        return isinstance(replacement, int) and not isinstance(replacement, bool)
    if isinstance(baseline, float):
        return isinstance(replacement, (int, float)) and not isinstance(replacement, bool)
    return type(baseline) is type(replacement)


def merge_sparse_override(baseline: Mapping[str, Any], override: Mapping[str, Any], *, solver: str | None = None) -> tuple[dict[str, Any], tuple[ConfigChangeRecord, ...]]:
    """Validate a sparse leaf override and return an independent effective config."""
    if not isinstance(override, Mapping):
        raise ValueError("Trial override root must be a mapping")
    leaves = flatten(override)
    effective = copy.deepcopy(dict(baseline)); changes: list[ConfigChangeRecord] = []
    for path in sorted(leaves):
        try: old = lookup(baseline, path)
        except KeyError:
            if solver is None or path not in OPTIONAL_DEFAULTS[solver]: raise ValueError(f"Unknown config path: {path}")
            old = OPTIONAL_DEFAULTS[solver][path]
        new = leaves[path]
        if not _same_type(old, new):
            raise ValueError(f"Type mismatch at {path}: expected {type(old).__name__}")
        if old == new:
            continue
        target: dict[str, Any] = effective
        parts = path.split(".")
        for part in parts[:-1]: target = target.setdefault(part, {})
        target[parts[-1]] = copy.deepcopy(new)
        changes.append(ConfigChangeRecord(path, old, new))
    # canonical_json also rejects non-finite numeric values.
    canonical_json(effective)
    return effective, tuple(changes)


def effective_config_identity(config: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def canonical_sparse_override(changes: Sequence[ConfigChangeRecord]) -> dict[str, Any]:
    """Reconstruct the normalized sparse nested representation from actual changes."""
    result: dict[str, Any] = {}
    for change in sorted(changes, key=lambda item: item.path):
        target = result
        parts = change.path.split(".")
        for part in parts[:-1]: target = target.setdefault(part, {})
        target[parts[-1]] = copy.deepcopy(change.new_value)
    return result


def protected_violations(solver: str, paths: Sequence[str]) -> tuple[PolicyViolationRecord, ...]:
    rules = PROTECTED[solver]
    return tuple(PolicyViolationRecord(path, "TRIAL.PROTECTED_PATH", "Trial overrides cannot change protected baseline identity/configuration")
                 for path in sorted(paths) if any(path == rule or path.startswith(rule + ".") for rule in rules))


def owner_stages(solver: str, path: str) -> tuple[str, ...] | None:
    found: list[str] = []
    for rule in OWNERSHIP[solver]:
        if _matches(path, rule.pattern): found.extend(rule.stages)
    return tuple(sorted(set(found))) or None


def stage_config_projection(solver: str, stage: str, effective: Mapping[str, Any]) -> dict[str, Any]:
    """The owned configuration leaves contributing to one producer signature."""
    leaves = flatten(effective)
    selected: dict[str, Any] = {}
    for path, value in leaves.items():
        owners = owner_stages(solver, path)
        if owners and stage in owners:
            selected[path] = value
    return selected


def action_config_projection(solver: str, stages: Sequence[str], effective: Mapping[str, Any]) -> dict[str, Any]:
    """Scientific config leaves consumed by one coarse, trusted action.

    A combined native call must bind every conceptual stage it actually
    executes.  Output-routing and other protected non-scientific fields have
    no ownership rule and are therefore deliberately excluded.
    """
    requested = set(stages)
    leaves = flatten(effective)
    return {path: value for path, value in leaves.items()
            if (owners := owner_stages(solver, path)) and requested.intersection(owners)}
