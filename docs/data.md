# Data

`data/` 模块承载图像观测、ROI 与数据集容器，以及种子/神经路径共享的图像填充
与 B-spline 预计算上下文。

## 文件与职责

```text
image.hpp                  Image 容器：包装 torch::Tensor，提供 width/height/channels
                           （构造与访问见 src/data/image.cpp）
roi.hpp                    ROI：单一连续区域（x_min,y_min,x_max,y_max），
                           contains(x,y) 与 validate()
dataset.hpp                DICDataset {reference: Image, deformed: Image, roi: ROI}
                           + validate()
stereo_dataset.hpp         StereoDataset {left: DICDataset, right: DICDataset}
multiview_dataset.hpp      MultiViewDataset {views: std::vector<DICDataset>}
image_precompute_context.hpp 图像填充与 B-spline 系数预计算：
                           ImagePrecomputeOptions {integer_search_radius,
                               coarse/fine_subset_radius, subset_radius,
                               bspline_border=3, bspline_degree=5}
                           calculate_image_padding(options)
                           mirror_pad_image(image, pad)
                           zero_pad_roi_mask(mask, pad)
                           ImagePrecomputeContext::create(...)（一次性预计算）
```

## 使用边界

- `Image` 是 C++ 问题装配的观测入口；张量布局（HWC/CHW）由绑定层约定，
  `image.cpp` 中的访问器统一返回逻辑尺寸。
- 预计算上下文（镜像填充 + B-spline 系数）在求解前一次性完成，属于非可微
  预处理；可微采样阶段只读这些固定系数（见 `interpolation.md`）。
- 数据集容器（单目/立体/多视角）只做校验与打包，不包含科学算法。
