# Problem

`problem/` 模块定义求解器消费的已装配输入（值对象）：数据、标定、B-spline
系数与初始化在此打包并校验；模型输出与坐标在其后变为可微。

## 文件与职责

```text
problem.hpp             基础接口 DICProblem {solver_type(), validate()}（抽象）
pin_problem.hpp         PINProblem（平面 2D PIN-DIC 值对象）：
                        校验后的参考/变形图像、掩码、B-spline 预计算上下文、
                        种子集、光度损失选项、MLP 模型选项
ndef_problem.hpp        NDeFProblem：多视角 NDeF 数据（相机模型、光度损失、
                        NDeFInternalModel 选项）；参考投影/可见性张量是固定
                        观测，绝不从变形图像推断
ndef_surface_problem.hpp NDeFSurfaceProblem：参考表面/深度网络问题
                        （稀疏 uv/camera/depth + 图像尺寸/ROI 掩码/查询点；
                        稠密参考图像细化输入 K/R/t/畸变/邻接拓扑；
                        预训练与稠密超参）
pin_stereo_problem.hpp  PINStereoProblem：三个平面 PIN 问题（参考视差/时间/
                        变形视差）+ 左右相机模型 + 重建选项 + world_scale
problem_builder.hpp     ProblemBuilder：把验证过的 data/calibration/
                        coefficients/init 组装成问题（build_pin_problem /
                        build_ndef_problem）
```

## 使用边界

- Problem 是**值对象**：只携带已校验输入，不做数值计算；`validate()` 在构造
  后立即调用。
- `PINStereoProblem` 是立体装配层：三份平面问题各自独立求解
  （`PINStereoSolver`），共享左右相机与重建配置。
- NDeF 问题的参考面投影/可见性属于固定观测，优化期间只读取。
- `DICProblem::solver_type()` 决定走 `PINSolver` 还是 `NDeFSolver`。
