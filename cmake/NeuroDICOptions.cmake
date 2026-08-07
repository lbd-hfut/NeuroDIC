option(NEURODIC_BUILD_TESTS "Build NeuroDIC C++ architecture tests" ON)
option(NEURODIC_BUILD_PYTHON "Build pybind11 Python extension neurodic._neurodic" ON)
option(NEURODIC_ENABLE_TORCH "Find LibTorch and build torch::Tensor implementation files" OFF)
option(NEURODIC_USE_OPENCV "Enable optional OpenCV preprocessing adapters" OFF)
option(NEURODIC_USE_EIGEN "Enable optional Eigen preprocessing helpers" OFF)
option(NEURODIC_USE_CERES "Enable Ceres bundle adjustment for OpenCV multiview calibration" ON)
option(NEURODIC_BUILD_OPENCV_CALIBRATION
    "Build the OpenCV/Eigen mono, stereo, and multiview calibration pipeline" ON)
