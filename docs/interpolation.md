# Interpolation

`interpolation/` 模块定义 NeuroDIC 唯一支持的插值族——B-spline：固定图像系数的
非可微预计算 + 可微采样。

## 文件与职责

```text
bspline.hpp              B-spline 次数校验：is_supported_bspline_degree /
                         validate_bspline_degree（支持 1/3/5，
                         core::InterpolationDegree {LINEAR, CUBIC, QUINTIC}）
bspline_coefficients.hpp 固定图像系数预处理：BSplineCoefficientBlock
                         {height, width, degree, pad_offset, coeff_cpu, ...}
                         在 torch::NoGradGuard 下计算（系数是固定观测），
                         degree 3/5 的递归滤波与边界条件见
                         src/interpolation/bspline_coefficients.cpp
torch_bspline.hpp        可微 LibTorch B-spline 采样器：在可微坐标处采样固定
                         系数，支持 CPU/CUDA，保留对坐标的 autograd
```

## 可微性契约

- **系数**：固定观测，`NoGradGuard` 下预计算，不参与梯度。
- **采样**：`torch_bspline` 的采样对坐标可微；神经场输出 → 采样 → 损失的整条
  路径必须保留 PyTorch autograd 图，禁止 NumPy/Eigen/OpenCV 往返。
- 验证：`tests/cpp/test_bspline.cpp` 与 `test_autograd_bspline.cpp` 覆盖基函数
  求值、系数布局与对坐标的梯度（有限差分 / gradcheck）。

## 使用边界

`ImagePrecomputeContext`（`data/`）持有填充图像与 `BSplineCoefficientBlock`，
供种子路径（整像素/亚像素搜索）与神经路径（可微光度采样）共用；神经路径只读
固定系数，不重新计算。
