#pragma once
#include "neurodic/problem/ndef_surface_problem.hpp"
#include "neurodic/core/result.hpp"
namespace neurodic { class NDeFSurfaceSolver { public: NDeFSurfaceResult solve(const NDeFSurfaceProblem&) const; }; }
