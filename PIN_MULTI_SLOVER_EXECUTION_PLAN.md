# `pin_multi_slover` 后续执行方案

## 定位与边界

`pin_multi_slover` 是一条独立的多相机 PIN-DIC 路线。它按选定的相机对运行，而不是把所有相机、参考曲面和可见性一起送入 NDeF 网络。该名称按需求保留 `slover` 拼写；代码类型使用 `PINMultiSolver`。

本次仅建立接口、配置、示例和测试占位，尚未接入 CMake、pybind11 或 `neurodic.__init__`。因此现有 PIN 2D、PIN stereo 和 NDeF 的代码路径、配置、输出目录和行为均不变。

## Traditional-DIC multi 路线参考

本路线的总体组织参考 `/home/a306/01project/Traditional-DIC` 的 multi-view 3D-DIC 思路，尤其是以下链路：

```text
相机对选择 → 参考时刻两两 SIFT 匹配生成 pair ROI
→ 每对 2D DIC → 每对 3D 重建（参考形貌、当前形貌、位移）
→ pairwise 曲面拼接/融合 → 形貌、位移场与应变后处理
```

对应参考实现主要位于 `Traditional-DIC/python/traditional_dic/multiview.py`：

- `generate_pair_masks_from_calibration()`：以两台相机的参考图 SIFT 匹配生成 pair-local ROI；
- `compute_pairwise_2d_dic()`：逐相机对执行二维相关；
- `compute_pairwise_3d_dic()`：以标定参数将每对结果分别重建为三维形貌与位移；
- `stitch_pairwise_3d_surfaces()`：完成 pairwise 三维产品的质量筛选、拼接与融合。

NeuroDIC 的 `pin_multi_slover` 将沿用这套“先 pairwise、后融合”的数据流，但二维场由现有 `PINSolver` 的三个 PIN 配准 (`A0→B0`、`A0→Ak`、`A0→Bk`) 产生。它不复用 Traditional-DIC 的具体代码，也不接入 NDeF 的稀疏轨迹/参考曲面 mask 流程。

## 目标数据流

```text
每个相机的参考图 A(t0), B(t0) ── SIFT 两两匹配 ──> pair-local ROI(A,B)
                                                     │
A(t0), B(t0), A(tk), B(tk) ── 三个 PIN 场 ──────────┤
  A0→B0, A0→Ak, A0→Bk                               │
                                                     v
                                      每对独立重建 X0、Xk 与 dX
                                                     │
                                                     v
                         多对曲面/位移场质量筛选、拼接与应变计算
```

## ROI 策略：必须与 NDeF 分开

1. 依据标定和相机布局选择相邻相机对；选择结果写入本路线自己的 manifest。
2. 只使用参考时刻的 `A(t0)` 与 `B(t0)` 提取/匹配 SIFT 特征；使用 ratio test、互检和 RANSAC 几何内点过滤。
3. 对每一对匹配点，在左相机参考像素坐标上构造 `convex_hull` 或 alpha-shape 支持域；可选腐蚀，生成该 pair 的左 ROI。右 ROI 也应同时保存以便质检和未来双向求解。
4. 输出固定为 `result/pin_multi_slover/pair_roi/<pair_id>/`，包含 `left_mask.npy/png`、`right_mask.npy/png`、匹配点、RANSAC 内点和 overlay。不得读取、复用、改写 `result/mask/per_camera/`。
5. 配对 ROI 失败（匹配数不足、RANSAC 失败或掩膜面积过小）应产生结构化的 skipped 记录，而不是悄悄退回 NDeF mask 或全图 mask。

NDeF 的 ROI 是根据已标定多视图稀疏观测与曲面可见性生成的每相机 mask；本路线的 ROI 是参考时刻的两两图像匹配支持域。两种策略、函数模块、配置段和输出根目录必须保持隔离。

## 分阶段落地

1. **配对与 ROI**：实现 `python/neurodic/pin_multi_roi.py`，写 pair manifest、掩膜和可视化；用合成图像和 CylinderDIC 的相机对测试匹配失败、遮挡和重复纹理。
2. **问题组装**：实现 `PINMultiProblem::validate()`，每对建立 `A0→B0`、`A0→Ak`、`A0→Bk` 三个 `PINProblem`，并明确左图网格、标定坐标系、尺度及时间索引。
3. **两两求解与重建**：复用已验证的 `PINSolver`，采用 stereo 路线相同的三场重建方式，输出每对 `X0`、`Xk`、`dX=Xk-X0`、有效掩膜和重投影误差。
4. **质量控制**：逐点检查图像边界、正深度、重投影误差、PIN 光度损失与匹配几何；以 pair 为单位保存原因码和统计，避免跨对融合时混入低质量结果。
5. **形貌与位移场融合**：先保存所有 pair 的独立产品；再实现可关闭的拼接层，按世界坐标、空间邻域、重投影误差和置信度去重/融合。刚体运动去除必须是显式配置。
6. **后处理与接口**：在融合点云/网格上计算形貌、`U/V/W`、位移模长和应变；随后才接入 CMake、pybind11、Python API、文档和端到端回归测试。

## 输出契约

建议固定目录如下，避免同名产物覆盖其他路线：

```text
result/pin_multi_slover/
  pair_roi/<pair_id>/
  pairs/<pair_id>/disp/
  pairs/<pair_id>/reconstruct/{reference,current}.npz
  pairs/<pair_id>/deformation/initial_to_current.npz
  fused/{reference_surface,current_surface,deformation}.npz
  manifest.json
visualization/pin_multi_slover/
  pair_roi/<pair_id>/
  pairs/<pair_id>/
  fused/
```

每个 `.npz` 必须附带世界坐标约定、`world_scale`、参考/当前帧、相机对、有效性标记和质量指标；融合产物还须保留来源 pair id，确保可追溯。

## 验收门槛

- 不修改或读取 NDeF mask 输出，且运行前后可对 NDeF 结果做哈希比对。
- 每个被执行的 pair 都有可复现的 ROI、匹配诊断、三张 2D PIN 场和两时刻 3D 重建结果。
- 合成刚体平移、已知位移和至少一个真实多相机 case 上，重投影误差、3D 位移误差及无效点比例均有报告。
- 仅在独立 pair 产品验证后，才启用 `fusion.enabled: true`；融合前后的点数、来源与剔除原因必须记录。
