# PINSolver

`solver/pin_solver.hpp`：`PINSolver` 是平面 2D PIN-DIC 的顶层求解器，通过一条
C++/LibTorch 路径完成 seed MSE 预训练 + SSD/ZNSSD 光度 Adam 优化。

```text
PINProblem
   ↓
Representation（PINDisplacementField / PINDisparityField）
   ↓
User-selectable Model（MLPModel，tanh MLP + 可选 Fourier 编码）
   ↓
Geometry（平面几何）
   ↓
B-spline sampling（torch_bspline，可微）
   ↓
Loss（PhotometricLoss: SSD / ZNSSD）
   ↓
Optimizer（Adam）
   ↓
PINResult
```

## PINStereoSolver

`solver/pin_stereo_solver.hpp`：`PINStereoSolver` 编排**三个已有 2D PIN 解**并做
CPU 立体重建（非可微后处理）：

- `solve(PINStereoProblem)`：依次求解
  `reference_disparity`（L0→R0）、`left_temporal`（L0→L1）、
  `deformed_disparity`（L0→R1）；
- `reconstruct(reference_disparity, left_temporal, deformed_disparity, problem)`：
  用 `StereoGeometry` / DLT 三角化重建参考与当前 3D 点，再计算 3D 位移。

对应 `PINProblem` 是平面值对象（`problem/pin_problem.hpp`），立体装配在
`PINStereoProblem` 层组合三份平面问题与左右相机模型。

命名约定：**不要新增** `PIN2DSolver` / `PINStereoSolver` 之外的
`pin_2d_solver.*` / `pin_stereo_solver.*` 文件；立体只是 `PINStereoProblem` +
`PINStereoSolver` 的组装。Python 入口为 `pin_dic()` / `pin_stereo_dic()`
（`python/neurodic/api/pin_dic.py`, `pin_stereo_dic.py`）。
