# Architecture

NeuroDIC 是 C++ 优先（C++-first）的 DIC 科学计算库：科学内核在 `include/neurodic`
与 `src`，LibTorch 提供可微路径，pybind11 生成 `neurodic._neurodic` 扩展，Python
包 `python/neurodic` 只做输入装配与结果编排，不重复实现科学内核。

```text
C++ Scientific Core (include/neurodic, src)
        ↓
Differentiable LibTorch Core (torch::Tensor, autograd)
        ↓
neurodic_core (CMake target)
        ↓
pybind11 (bindings/python)
        ↓
neurodic._neurodic
        ↓
Thin Python API (python/neurodic)
```

## 模块清单

`include/neurodic` 下的模块（对应 `src` 下的实现与 `bindings/python/bind_*.cpp`）：

```text
core            枚举/配置/上下文/异常/随机/结果（types, config, context, exceptions, random, result）
data            图像/ROI/数据集容器与预计算上下文（image, roi, dataset, stereo_dataset, multiview_dataset, image_precompute_context）
interpolation   B-spline 系数与可微采样（bspline, bspline_coefficients, torch_bspline）
initialization  稀疏 warm-start 预处理（initializer, integer_search, sift_*, subpixel_search, seed_set, seed_cleanup, sparse_prior, sampling, ndef_precalculation, output_normalization）
calibration     相机模型与标定（camera_model, calibration_result, calibration_manager, mono/stereo/multiview_calibration, colmap_calibration）
problem         求解器输入装配（problem, pin_problem, ndef_problem, ndef_surface_problem, pin_stereo_problem, problem_builder）
representation  神经场输出解码为物理场（representation, pin_displacement_field, pin_disparity_field, ndef_deformation_field, ndef_surface_field）
model           神经模型（neural_model, mlp, fourier, ndef_internal_model, ndef_depth_model, model_factory, resnet, siren）
geometry        几何引擎（geometry, projection, triangulation, stereo_geometry, ndef_geometry, visibility, coordinate_transform）
loss            可微目标（loss, mse, ssd, znssd, photometric, regularization）
optimizer       优化循环（optimizer, adam, lbfgs, convergence, scheduler）
solver          顶层求解器（solver, pin_solver, pin_stereo_solver, ndef_solver, ndef_surface_solver）
postprocess     求解后处理（displacement, filtering, physical_scale, strain）
```

## Solver Families

```text
Solver
├── PINSolver           平面 2D PIN-DIC（seed MSE 预训练 + SSD/ZNSSD 光度 Adam 优化）
├── PINStereoSolver     组合三个 2D PIN 解（参考视差/时间/变形视差）+ CPU 立体重建
├── NDeFSolver          多视角 NDeF-DIC（内部受控模型拓扑）
└── NDeFSurfaceSolver   参考表面/深度网络预训练与稠密表面细化
```

`PINSolver` 同时服务 PIN-DIC 2D 与 PIN-DIC Stereo：立体专属行为由 data、
calibration、geometry、problem、result 层承载，不在 solver 内分叉。`NDeFSolver`
拥有内部受控的模型拓扑（不暴露任意层数/宽度/跳连配置）。

## Problem Flow

```text
Data + ROI + Calibration + B-spline coefficients + Initialization
        ↓
ProblemBuilder
        ↓
PINProblem / NDeFProblem / NDeFSurfaceProblem / PINStereoProblem
        ↓
PINSolver / NDeFSolver / NDeFSurfaceSolver / PINStereoSolver
        ↓
PINResult / NDeFResult / NDeFSurfaceResult / PINStereoResult
```

标定与求解器解耦：`CalibrationResult` 只被 problem builder 消费，solver 不解析
标定。一个 ROI 对应一个连续神经场；MSPINN/FBPINN 多区域域分解有意不实现。

## 可微性规则

- 模型输出到损失之间的一切运算必须走 `torch::Tensor` 并保留 autograd 图，
  禁止 NumPy/Eigen/OpenCV 往返（`differentiable_core.md` 详述）。
- 初始化、标定、预计算（B-spline 系数、种子）属于非可微预处理，可自由使用
  OpenCV/Eigen 辅助实现。
- 绑定层只暴露经过验证的接口，不自动暴露每个内部类。
