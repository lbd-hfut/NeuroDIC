"""Foundational, solver-free contracts for NeuroDIC's agent control plane.

This package intentionally contains no CLI, inspection, evaluation, or solver
orchestration.  Those capabilities build on the records defined here.
"""

from .artifacts import (
    ArtifactRecord,
    IdentityRecord,
    IdentityStrength,
    canonical_path,
    content_identity,
    metadata_identity,
    path_within,
    require_path_within,
)
from .errors import ControlPlaneError, ErrorRecord, error_envelope
from .schemas import (
    AGENT_SCHEMA_VERSION,
    QUALITY_SCHEMA_VERSION,
    Availability,
    CapabilityRecord,
    Envelope,
    QualityReport,
    MetricRecord, ThresholdResult, FindingRecord,
    canonical_json,
    is_utc_timestamp,
    new_id,
    utc_now,
)
from .state import RunRecord, StageRecord, StageStatus, TrialRecord, atomic_write_json, valid_stage_transition
from .trials import TrialPlan, plan_trial
from .execution import ProducerSignature, TrustedAction, execute_trial
from .compare import compare_quality_reports, compare_results, quality_identity, select_best_candidate
from .best import evaluate_best_candidate, load_best, update_best
from .parameters import InterventionRule, ParameterMetadata, load_intervention_rules, load_parameter_registry
from .recommend import ParameterChangeRecommendation, RecommendationReport, diagnosis_identity, recommend_from_diagnosis

__all__ = [
    "AGENT_SCHEMA_VERSION", "QUALITY_SCHEMA_VERSION", "ArtifactRecord", "Availability",
    "CapabilityRecord", "ControlPlaneError", "Envelope", "ErrorRecord", "IdentityRecord",
    "IdentityStrength", "QualityReport", "MetricRecord", "ThresholdResult", "FindingRecord", "RunRecord", "StageRecord", "StageStatus", "TrialRecord",
    "atomic_write_json", "canonical_json", "canonical_path", "content_identity", "error_envelope",
    "is_utc_timestamp", "metadata_identity", "new_id", "path_within", "require_path_within", "utc_now",
    "valid_stage_transition",
    "TrialPlan", "plan_trial",
    "ProducerSignature", "TrustedAction", "execute_trial",
    "compare_quality_reports", "compare_results", "quality_identity", "select_best_candidate", "evaluate_best_candidate", "load_best", "update_best",
    "InterventionRule", "ParameterMetadata", "ParameterChangeRecommendation", "RecommendationReport", "load_intervention_rules", "load_parameter_registry", "diagnosis_identity", "recommend_from_diagnosis",
]
