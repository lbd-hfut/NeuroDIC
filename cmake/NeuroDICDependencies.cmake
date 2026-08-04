set(NEURODIC_TORCH_FOUND OFF)
if(NEURODIC_ENABLE_TORCH)
    find_package(Torch REQUIRED)
    if(Torch_FOUND)
        set(NEURODIC_TORCH_FOUND ON)
        # pybind11 bindings that expose torch::Tensor need PyTorch's Python
        # type casters, which live in libtorch_python rather than libtorch.
        find_library(NEURODIC_TORCH_PYTHON_LIBRARY torch_python
            HINTS "${TORCH_INSTALL_PREFIX}/lib" "${CMAKE_PREFIX_PATH}/lib")
        message(STATUS "NeuroDIC: found LibTorch at ${Torch_DIR}")
    endif()
else()
    message(STATUS
        "NeuroDIC: NEURODIC_ENABLE_TORCH=OFF. LibTorch is architecturally "
        "required for implementation builds; enable it with "
        "-DNEURODIC_ENABLE_TORCH=ON and provide Torch_DIR or CMAKE_PREFIX_PATH.")
endif()

set(NEURODIC_PYBIND11_FOUND OFF)
# pybind11 (>= 3.0, NewTools) skips find_package(Python) when Python_FOUND is
# already set (e.g. by Torch's Caffe2 config) and relies on python_add_library
# being defined by an earlier explicit find_package(Python). Without this,
# pybind11_add_module fails with "Unknown CMake command python_add_library".
find_package(Python 3.8 COMPONENTS Interpreter Development.Module QUIET)
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
