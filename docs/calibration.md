# Calibration

标定与求解器解耦：求解器只消费 `CalibrationResult`，不检测标定板、不解析 COLMAP、
不估计相机参数。

```text
Mono / Stereo / COLMAP calibration
        ↓
CalibrationResult
        ↓
ProblemBuilder
        ↓
Problem
        ↓
Solver
```

## calibration/ 模块

```text
camera_model.hpp        CameraModel {intrinsics[3,3], rotation[3,3](world->camera),
                       translation[3], distortion[N], image_width/height, rms_error,
                       label}；validate() 与 projection_matrix()（K[R|t], [3,4]）
calibration_result.hpp  CalibrationResult {type: NONE/MONO/STEREO/COLMAP, cameras,
                       stereo_rotation/translation（左->右，可选）, rms_error}
calibration_manager.hpp CalibrationManager::calibrate(CalibrationType)
colmap_calibration.hpp  COLMAPCalibration：解析 COLMAP 输出为 CalibrationResult
mono_calibration.hpp    Mono 标定接口/占位
stereo_calibration.hpp  Stereo 标定入口与结果
multiview_calibration.hpp 多视角（传统）自标定：MultiviewCalibrationOptions /
                       MultiviewCalibrationResult / MultiviewStageStat /
                       MultiviewRegistrationAttempt / SparsePoint3D /
                       SparsePointDiagnostic / 棋盘格尺度选项与结果
```

## 传统多视角自标定（colmap_like）

`src/calibration/traditional/multiview_calibration.cpp` 实现 COLMAP 风格增量
重建（详见下方"与官方 COLMAP 的实质差异"）：

1. 匹配图：exhaustive 全对几何验证匹配（`matching_mode: exhaustive`）或
   legacy 窗口匹配（`window`）。
2. 初始像对：COLMAP 排序 + 几何检查（前向运动、中位三角化角）+ 初始 BA
   可持续性 + 首个增量注册存活验证，候选耗尽时按 COLMAP 语义松弛阈值。
3. 增量注册：候选按可见点数排名，PnP 失败记录原因并继续；注册后局部细化
   （Cauchy 鲁棒损失），全局细化（TRIVIAL）按帧/点增长触发。
4. 三角化/过滤：create/continue 角误差准则、complete/merge/retriangulate，
   最终通用几何过滤（`final_min_track_length`、`final_max_depth_ratio`）。
5. 共享 SIMPLE_PINHOLE：注册/重建期间内参固定，可选最终仅焦距全局 BA
   （`final_refine_focal_length`）。
6. 诊断：`MultiviewStageStat`（每阶段相机/点数/观测/RMS/内参）、
   `MultiviewRegistrationAttempt`（每次注册尝试及失败原因）、
   `SparsePointDiagnostic`（每点创建来源/逐观测误差/三角化角/正深度/BA 前后状态）。

## Python 装配（python/neurodic/calibration.py）

- `calibrate_multiview_colmap_like` / `run_multiview_case`：调用 C++ 后端并序列化
  结果（`calibration_result.json`、`observations.npz`、`camera_pairs.json`、
  `calibration_scale.json`、点级诊断）。
- 尺度恢复：优先棋盘格三角化统一尺度，失败回退元相机模型 Sim(3) 对齐
  （`estimate_multiview_chessboard_scale` / `_align_sfm_to_metric_cameras`）。
- 配置：`config/calibration.yaml`（`outputs.result_subdir` 控制结果子目录）。

## 与官方 COLMAP 的实质差异

NeuroDIC 的 `multiview_calibration`（`src/calibration/traditional/multiview_calibration.cpp`）
以 COLMAP `incremental_mapper` / `incremental_triangulator` 语义为对齐目标，但仍是
独立实现。以下是与官方 COLMAP / PyCOLMAP 相比仍未消除的实质差异（截至 2026-08，
CylinderDIC 案例下 12/12 相机、2214 个 3-view+ 稀疏点、径向 p01–p99 = 79.90–80.31）。

### 1. SIFT 特征提取器

- COLMAP 使用自带 VLFeat SIFT（`SiftExtraction`），支持 Upright SIFT、自定义
  peak/edge 阈值与更大的默认 `max_num_features = 16384`。
- NeuroDIC 使用 OpenCV `cv::SIFT`（`max_features`、`sift_contrast_threshold` 可配，
  RootSIFT 开关 `root_sift`）。同一场景下 COLMAP 的每对验证内点约为 OpenCV 的
  1.3 倍，最终 3-view+ 点数约为 COLMAP 的 76%（经 `sift_contrast_threshold=0.02`
  受控实验校准；RootSIFT 对照实验在本案例上退化，未采纳）。
- 影响：点数规模与匹配池大小，不影响几何管线正确性。

### 2. 初始像对选择

- COLMAP：`FindFirstInitialImage`/`FindSecondInitialImage` 按对应数排序后逐个
  `TryInitializeForTwoViewGeometry`，坏对在初始化阶段失败后被跳过（外层
  `num_trials` 循环 + `init_image_pairs` 去重）。
- NeuroDIC：相同排序与几何检查（前向运动、中位三角化角 ≥ `init_min_tri_angle`），
  并额外要求**首个增量注册存活验证**——候选初始化可持续后必须成功注册一个增量
  相机且局部细化后点数 ≥ `abs_pose_min_num_inliers`，否则拒绝并尝试下一候选。
  这等效于 COLMAP 的 `ReconstructSubModel` 失败回退，但把"模型坍塌"检测提前到
  第一个增量注册，避免为坏对跑完整增量。代价是每个候选多一次增量注册试算。
- `init_min_tri_angle_degrees` 默认 45°（COLMAP 16°）：弱基线相邻对在环形数据上
  必然漂移（基线坍塌），更高阈值让候选偏好宽基线种子；候选耗尽时仍按 COLMAP
  语义逐级松弛（`/2`）。

### 3. 图像选择 / 注册顺序

- COLMAP 默认 `image_selection_method = MIN_UNCERTAINTY`（倾向姿态不确定的相机），
  NeuroDIC 使用 max-visible-points 排名（`rank_next_images`）。两者在环上都会产生
  交替增长，结果相当；未移植不确定性度量。
- 候选注册失败记录 `MultiviewRegistrationAttempt`（可见点数、PnP 对应数/内点、
  原因），失败后继续尝试下一候选，与 COLMAP 一致。

### 4. 两视图几何分类（TwoViewGeometry）

- COLMAP 的 `TwoViewGeometry::Estimate` 显式分类 PLANAR / PANO / CALIBRATED /
  DEGENERATE，DEGENERATE 直接拒绝进入初始对。
- NeuroDIC 未实现该分类，改用两个等效手段：强基线偏好（第 2 节）与首个增量注册
  存活验证。二者联合能自动拒绝"匹配多但几何质量差"的候选对（如 CylinderDIC 的
  (8,6)）。

### 5. 结构无注册回退（structure-less registration）

- COLMAP `RegisterNextStructureLessImage` 使用 2D-2D 极线几何最小解算器（官方
  estimators 库），该解算器不在 NeuroDIC 代码库内，**未移植**。
- 配置项 `structure_less_registration_fallback: false` 显式关闭；PnP 失败的候选
  不中断重建，记录原因并等待全局细化后重试。

### 6. 三角化与过滤

- COLMAP `IncrementalTriangulator` 的 create/continue 用角误差 + RANSAC
  （`EstimateTriangulation`）；NeuroDIC 用线性 DLT + 角误差内点准则，语义一致。
- `ignore_two_view_tracks` 语义一致（候选列表上检查，非 inlier 后）。
- NeuroDIC 额外提供**最终通用几何过滤**（`final_min_track_length=3`、
  `final_max_depth_ratio=4.0`，逐观测误差/正深度/最小三角化角）：一次 BA 后清理
  不可多视图验证的 2-view 点（实验证明 2-view 点是极端径向离群的主要来源）。
  该过滤不使用任何真值半径，是通用几何条件；COLMAP 的 `FilterPoints` 不按 track
  长度过滤（其 2-view 点质量更高，源于更干净的匹配）。

### 7. Bundle Adjustment

- 结构与 COLMAP 一致：Ceres SPARSE_SCHUR、共享 SIMPLE_PINHOLE、双相机 gauge
  （anchor 位姿常量 + 第二相机旋转自由/单平移分量固定）、局部 BA（Cauchy 鲁棒
  损失）→ 全局 BA（TRIVIAL），全局细化后 `Reconstruction::Normalize`。
- 未复刻 COLMAP 的梯度/函数容差默认值与损失函数尺度微调（`gradient_tolerance`、
  `ba_global_function_tolerance` 等）；收敛行为近似但不逐项一致。
- 最终焦距 BA（`final_refine_focal_length`）为 NeuroDIC 对 CylinderDIC 基准流程
  的显式阶段，COLMAP 无直接对应选项。

### 8. 点级可追溯诊断

- NeuroDIC 为每个稀疏点输出 `SparsePointDiagnostic`（point id、track 长度、观测
  相机、逐观测误差、三角化角、正深度、创建来源 create/merge/retriangulate、
  最终 BA 前后坐标与 RMS、最终过滤结果），COLMAP 无等价运行时输出。
