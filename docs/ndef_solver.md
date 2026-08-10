# NDeFSolver

`solver/ndef_solver.hpp`：`NDeFSolver` 拥有 NDeF 多视角 DIC 求解流程，并控制其
内部模型拓扑（`NDeFInternalModel` 的层数/宽度/编码器由求解器决定，不对外暴露）。

```text
NDeFProblem
   ↓
Reference Surface Representation（NDeFSurfaceField）
   ↓
NDeF Deformation Representation（NDeFDeformationField）
   ↓
Internal Model（NDeFInternalModel, torch::nn::Module）
   ↓
NDeFGeometry（多视角投影 + 可见性）
   ↓
Photometric Loss
   ↓
Optimizer（Adam，seed MSE 预训练 + 光度优化）
   ↓
NDeFResult（SolverDiagnostics + 场张量）
```

## NDeFSurfaceSolver

`solver/ndef_surface_solver.hpp`：`NDeFSurfaceSolver` 面向参考表面/深度网络
（`NDeFDepthModel`，C++/LibTorch 版 SfMDepthFiLMNet）：

- `solve(NDeFSurfaceProblem)`：稀疏预训练 + 稠密参考图像细化
  （`dense_iterations` / `dense_epochs` / `dense_samples_per_camera`，
  auto-batch 模式与 Python NDeF-DIC scheduler 一致）。
- 输入张量：稀疏 uv / camera / depth、图像尺寸、ROI 掩码、查询 uv/camera，
  以及稠密细化用的 reference_images、内参 K/R/t、畸变与邻接拓扑
  `dense_neighbors[V,2]`（无拓扑邻居记为 -1）。

Python API `ndef_dic()` / `pretrain_ndef_surface()`（`python/neurodic/api/ndef_dic.py`、
`ndef_surface.py`）装配 `config/ndef_multi.yaml` 并调用绑定；公共 Python/YAML
接口不得暴露任意 NDeF 层数、宽度、跳连拓扑或内部分支。
