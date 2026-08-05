option(NEURODIC_BUILD_TESTS "Build NeuroDIC C++ architecture tests" ON)
option(NEURODIC_BUILD_PYTHON "Build pybind11 Python extension neurodic._neurodic" ON)
option(NEURODIC_ENABLE_TORCH "Find LibTorch and build torch::Tensor implementation files" OFF)
option(NEURODIC_USE_OPENCV "Enable optional OpenCV preprocessing adapters" OFF)
option(NEURODIC_USE_EIGEN "Enable optional Eigen preprocessing helpers" OFF)
option(NEURODIC_BUILD_TRADITIONAL_CALIBRATION
    "Build the Traditional-DIC mono/stereo/multiview calibration port" ON)
