#include <cassert>
#include <iostream>

#include "neurodic/core/context.hpp"
#include "neurodic/core/types.hpp"

void test_roi();
void test_bspline();
void test_autograd_bspline();
void test_initialization();
void test_geometry();
void test_solver_interfaces();
void test_pin_solver();

int main() {
    static_assert(static_cast<int>(neurodic::InterpolationDegree::LINEAR) == 1);
    static_assert(static_cast<int>(neurodic::InterpolationDegree::CUBIC) == 3);
    static_assert(static_cast<int>(neurodic::InterpolationDegree::QUINTIC) == 5);

    neurodic::RuntimeContext context;
    context.validate();

    test_roi();
    test_bspline();
    test_autograd_bspline();
    test_initialization();
    test_geometry();
    test_solver_interfaces();
    test_pin_solver();

    std::cout << "NeuroDIC C++ architecture tests passed\n";
    return 0;
}
