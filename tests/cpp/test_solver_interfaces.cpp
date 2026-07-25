#include <type_traits>

#include "neurodic/solver/ndef_solver.hpp"
#include "neurodic/solver/pin_solver.hpp"

void test_solver_interfaces() {
    static_assert(!std::is_same_v<neurodic::PINSolver, neurodic::NDeFSolver>);
}
