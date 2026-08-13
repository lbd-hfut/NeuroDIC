cmake_minimum_required(VERSION 3.24)

project(NeuroDIC VERSION 0.1.0 LANGUAGES CXX C)

include(cmake/NeuroDICOptions.cmake)
include(cmake/NeuroDICCompilerWarnings.cmake)
include(cmake/NeuroDICDependencies.cmake)

set(NEURODIC_OPENCV_CALIBRATION_SOURCES)
if(NEURODIC_BUILD_OPENCV_CALIBRATION)
    list(APPEND NEURODIC_OPENCV_CALIBRATION_SOURCES
        src/calibration/mono_calibration.cpp
        src/calibration/stereo_calibration.cpp
        src/calibration/multiview_calibration.cpp
    )
endif()

if(NEURODIC_TORCH_FOUND)
    add_library(neurodic_core
        src/core/result.cpp
        src/core/context.cpp
        src/core/random.cpp
        src/data/roi.cpp
        src/data/image.cpp
        src/data/image_precompute_context.cpp
        src/data/multiview_dataset.cpp
        src/calibration/camera_model.cpp
        src/calibration/calibration_result.cpp
        ${NEURODIC_OPENCV_CALIBRATION_SOURCES}
        src/geometry/projection.cpp
        src/geometry/triangulation.cpp
        src/geometry/stereo_geometry.cpp
        src/geometry/ndef_geometry.cpp
        src/interpolation/bspline.cpp
        src/interpolation/bspline_coefficients.cpp
        src/interpolation/torch_bspline.cpp
        src/loss/mse.cpp
        src/loss/ssd.cpp
        src/loss/znssd.cpp
        src/loss/photometric.cpp
        src/initialization/seed_set.cpp
        src/initialization/seed_cleanup.cpp
        src/initialization/sift_grid_seed_initializer.cpp
        src/initialization/traditional_seed_initializer.cpp
        src/initialization/ndef_precalculation.cpp
        src/model/fourier.cpp
        src/model/mlp.cpp
        src/model/ndef_internal_model.cpp
        src/model/ndef_depth_model.cpp
        src/model/model_factory.cpp
        src/optimizer/adam.cpp
        src/postprocess/strain.cpp
        src/postprocess/filtering.cpp
        src/postprocess/surface_mesh.cpp
        src/postprocess/surface_field.cpp
        src/representation/pin_displacement_field.cpp
        src/representation/ndef_surface_field.cpp
        src/representation/ndef_deformation_field.cpp
        src/problem/pin_problem.cpp
        src/problem/pin_stereo_problem.cpp
        src/problem/pin_multi_problem.cpp
        src/problem/ndef_problem.cpp
        src/problem/ndef_surface_problem.cpp
        src/solver/pin_solver.cpp
        src/solver/pin_stereo_solver.cpp
        src/solver/pin_multi_slover.cpp
        src/solver/ndef_solver.cpp
        src/solver/ndef_surface_solver.cpp
    )
    # The static core is linked into the pybind11 shared module (_neurodic),
    # so all objects must be position-independent.
    set_target_properties(neurodic_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
    add_library(neurodic_core INTERFACE)
endif()

if(NEURODIC_TORCH_FOUND)
    target_include_directories(neurodic_core
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_compile_features(neurodic_core PUBLIC cxx_std_17)
    neurodic_apply_warnings(neurodic_core)
    target_link_libraries(neurodic_core PUBLIC ${TORCH_LIBRARIES})
    target_compile_definitions(neurodic_core PUBLIC NEURODIC_HAS_TORCH=1)
    if(NEURODIC_BUILD_OPENCV_CALIBRATION)
        target_link_libraries(neurodic_core PUBLIC Eigen3::Eigen)
        target_compile_definitions(neurodic_core PUBLIC NEURODIC_HAS_OPENCV_CALIBRATION=1)
    endif()
    if(NEURODIC_CERES_FOUND)
        target_link_libraries(neurodic_core PUBLIC Ceres::ceres)
        target_compile_definitions(neurodic_core PUBLIC NEURODIC_HAS_CERES=1)
    endif()
    if(OpenCV_FOUND)
        target_link_libraries(neurodic_core PUBLIC ${OpenCV_LIBS})
        target_include_directories(neurodic_core PUBLIC ${OpenCV_INCLUDE_DIRS})
        target_compile_definitions(neurodic_core PUBLIC NEURODIC_HAS_OPENCV=1)
    endif()
else()
    target_include_directories(neurodic_core
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
endif()

if(NEURODIC_BUILD_PYTHON AND NEURODIC_PYBIND11_FOUND)
    pybind11_add_module(_neurodic
        bindings/python/module.cpp
        bindings/python/bind_core.cpp
        bindings/python/bind_data.cpp
        bindings/python/bind_interpolation.cpp
        bindings/python/bind_initialization.cpp
        bindings/python/bind_calibration.cpp
        bindings/python/bind_opencv_calibration.cpp
        bindings/python/bind_problem.cpp
        bindings/python/bind_geometry.cpp
        bindings/python/bind_solver.cpp
        bindings/python/bind_result.cpp
        bindings/python/bind_postprocess.cpp
    )
    target_link_libraries(_neurodic PRIVATE neurodic_core)
    if(NEURODIC_TORCH_PYTHON_LIBRARY)
        target_link_libraries(_neurodic PRIVATE ${NEURODIC_TORCH_PYTHON_LIBRARY})
    endif()
    set_target_properties(_neurodic PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/neurodic
    )
    install(TARGETS _neurodic LIBRARY DESTINATION neurodic)
endif()

if(NEURODIC_BUILD_TESTS AND NEURODIC_TORCH_FOUND)
    enable_testing()
    add_executable(neurodic_cpp_tests
        tests/cpp/test_core.cpp
        tests/cpp/test_random.cpp
        tests/cpp/test_roi.cpp
        tests/cpp/test_bspline.cpp
        tests/cpp/test_autograd_bspline.cpp
        tests/cpp/test_initialization.cpp
        tests/cpp/test_geometry.cpp
        tests/cpp/test_solver_interfaces.cpp
        tests/cpp/test_pin_solver.cpp
        tests/cpp/test_pin_stereo_solver.cpp
        tests/cpp/test_pin_multi_solver.cpp
        tests/cpp/test_ndef_solver.cpp
        tests/cpp/test_ndef_precalculation.cpp
        tests/cpp/test_postprocess.cpp
    )
    target_link_libraries(neurodic_cpp_tests PRIVATE neurodic_core)
    neurodic_apply_warnings(neurodic_cpp_tests)
    add_test(NAME neurodic_cpp_tests COMMAND neurodic_cpp_tests)
elseif(NEURODIC_BUILD_TESTS)
    enable_testing()
    message(STATUS "NeuroDIC: skipping C++ tests because LibTorch implementation build is disabled/unavailable.")
    add_test(
        NAME neurodic_dependency_report
        COMMAND ${CMAKE_COMMAND} -E echo "LibTorch implementation build disabled/unavailable; C++ torch tests skipped."
    )
endif()
