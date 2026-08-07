# Initialization

初始化属于**非可微预处理**：在模型-损失 autograd 路径之外，可自由使用
OpenCV/Eigen 辅助实现。

```text
ROI
 ↓
uniform sampling / SIFT（sift_initializer, sift_grid_seed_initializer, traditional_seed_initializer）
 ↓
integer-pixel 搜索 / subpixel 细化（integer_search, subpixel_search）
 ↓
SparsePrior / SeedSet（seed_set, seed_cleanup, sparse_prior）
 ↓
输出归一化 mean/scale（output_normalization, ndef_precalculation）
 ↓
神经场 warm start（InitializationResult）
```

## 文件与职责

```text
initializer.hpp            接口 Initializer::run() -> InitializationResult
initialization_result.hpp  InitializationResult {coordinates, displacement, confidence}
integer_search.hpp         IntegerSearchInitializer：整像素粗对应估计
subpixel_search.hpp        SubpixelSearch：神经训练前的亚像素细化
sift_initializer.hpp       SIFTInitializer：特征点稀疏 warm start
sift_grid_seed_initializer.hpp 网格化 SIFT 种子初始化（PIN-DIC）
traditional_seed_initializer.hpp 传统（灰度/窗口）种子初始化（PIN-DIC）
seed_set.hpp               SeedSet {seed_pos[N,2], seed_uv[N,2], scale_uv[4]}
                           （原图坐标；含 empty/constant/from_tensors 构造）
seed_cleanup.hpp           SeedCleanupOptions {mad_threshold=4.5, min_seed_count=3}
                           clean_seed_set：MAD 离群剔除 + 由保留种子推 scale_uv
sparse_prior.hpp           SparsePrior：稀疏位移先验容器
output_normalization.hpp   estimate_output_normalization(prior)：位移 mean/scale 估计
sampling.hpp               单 ROI 内采样坐标生成
ndef_precalculation.hpp    稳健 NDeF-DIC 稀疏位移尺度预处理（NDeFDisplacementScale：
                           inlier_mask/median/mean/p75/p90/maximum）
```

种子输出约定：`SeedSet` 的坐标/位移使用**原始图像坐标 (x, y)**，`scale_uv`
保存 (mean_u, mean_v, halfrange_u, halfrange_v) 供输出归一化。
