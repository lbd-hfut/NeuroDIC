# C++ Core

`neurodic_core` 是科学内核的 CMake target：链接 LibTorch（可用时）与 Ceres（启用
`NEURODIC_USE_CERES` 的传统标定路径），是 pybind11 扩展 `neurodic._neurodic` 的
依赖。公共接口在 `include/neurodic`，实现（含占位骨架）在 `src`。

## core/ 模块内容

```text
core/types.hpp      架构级强类型枚举：
                     SolverType { PIN, NDEF }
                     CalibrationType { NONE, MONO, STEREO, COLMAP }
                     GeometryType { PLANAR_2D, STEREO, NDEF_MULTIVIEW }
                     InterpolationDegree { LINEAR=1, CUBIC=3, QUINTIC=5 }
                     SolverStatus { NOT_STARTED, RUNNING, CONVERGED, NOT_CONVERGED, FAILED }
core/config.hpp     Config 外壳（name + validate()），未承诺最终 schema
core/context.hpp    RuntimeContext：device（"auto"）、dtype（默认 kFloat32）、
                    随机种子、debug 开关；validate() 在 src/core/context.cpp
core/exceptions.hpp 异常层次：NeuroDICError -> ValidationError /
                     NotImplementedScientificError
core/random.hpp     set_random_seed(seed)：进程级确定性随机状态（LibTorch +
                    可选 OpenCV 全局 RNG）
core/result.hpp     SolverDiagnostics（状态/迭代数/最终损失/指标 map）与求解器
                    结果张量容器（PINResult/NDeFResult/NDeFSurfaceResult/
                    PINStereoResult 等在 bind_result.cpp 中绑定）
```

## 结构约定

各模块（data, interpolation, initialization, calibration, problem,
representation, model, geometry, loss, optimizer, solver, postprocess）遵循
统一的头文件注释契约：

```text
Responsibilities: 模块职责
Inputs:            输入
Outputs:           输出
Ownership:         所有权约定
Differentiable:    是否处于可微路径（NO / YES / PARTIAL）
```

科学方法要么实现为经过验证的数值算法，要么抛 `NotImplementedScientificError` /
保持抽象，**不允许返回编造数值**。
