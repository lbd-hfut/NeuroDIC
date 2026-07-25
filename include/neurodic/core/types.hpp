/**
 * NeuroDIC core enum definitions.
 *
 * Responsibilities: define stable architecture-level tags shared by data,
 * problem, solver, and binding layers.
 * Inputs: none.
 * Outputs: strongly typed enum values.
 * Ownership: value types only.
 * Differentiable: NO.
 * TODO(NeuroDIC): keep public enum growth conservative as APIs stabilize.
 */
#pragma once

namespace neurodic {

enum class SolverType { PIN, NDEF };
enum class CalibrationType { NONE, MONO, STEREO, COLMAP };
enum class GeometryType { PLANAR_2D, STEREO, NDEF_MULTIVIEW };
enum class InterpolationDegree : int { LINEAR = 1, CUBIC = 3, QUINTIC = 5 };
enum class SolverStatus { NOT_STARTED, RUNNING, CONVERGED, NOT_CONVERGED, FAILED };

}  // namespace neurodic
