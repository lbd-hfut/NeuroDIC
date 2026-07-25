set(NEURODIC_TORCH_FOUND OFF)
if(NEURODIC_ENABLE_TORCH)
    find_package(Torch REQUIRED)
    if(Torch_FOUND)
        set(NEURODIC_TORCH_FOUND ON)
        message(STATUS "NeuroDIC: found LibTorch at ${Torch_DIR}")
    endif()
else()
    message(STATUS
        "NeuroDIC: NEURODIC_ENABLE_TORCH=OFF. LibTorch is architecturally "
        "required for implementation builds; enable it with "
        "-DNEURODIC_ENABLE_TORCH=ON and provide Torch_DIR or CMAKE_PREFIX_PATH.")
endif()

set(NEURODIC_PYBIND11_FOUND OFF)
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    set(NEURODIC_PYBIND11_FOUND ON)
    message(STATUS "NeuroDIC: found pybind11")
elseif(NEURODIC_BUILD_PYTHON)
    message(WARNING
        "NeuroDIC: pybind11 was not found. neurodic._neurodic will not be built. "
        "Install pybind11 or pass pybind11_DIR to enable Python bindings.")
endif()

if(NEURODIC_USE_OPENCV)
    find_package(OpenCV QUIET)
endif()

if(NEURODIC_USE_EIGEN)
    find_package(Eigen3 QUIET)
endif()
