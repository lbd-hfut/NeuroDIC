# Loss

`loss/` 模块定义可微标量目标：基础接口 + 具体损失（MSE / SSD / ZNSSD / 光度
选择器）+ 正则化。

## 文件与职责

```text
loss.hpp            接口 Loss::compute(residual) -> torch::Tensor（可微标量）
mse.hpp             MSELoss：种子监督用均方误差
ssd.hpp             SSDLoss：光度 sum-of-squared-differences
                    compute(residual)
                    compute(reference, deformed)
                    compute_masked(reference, deformed, mask)
znssd.hpp           ZNSSDLoss：零均值归一化 SSD（PhotometricLoss 默认）
                    ZNSSDLossOptions {epsilon=1e-6, kernel_size=7}
                    compute(residual) / compute(reference, deformed) /
                    compute_windows(...)
photometric.hpp     PhotometricLoss 选择器：PhotometricLossType {SSD, ZNSSD}
                    PhotometricLossOptions {type, znssd}
regularization.hpp  regularization_term(field)：平滑/物理先验外壳
```

## 使用边界

- 所有 `compute` 在残差或图像张量上操作并保留 autograd；损失位于模型输出与
  优化器之间。
- PIN 路径：seed MSE 预训练 + `PhotometricLoss`（SSD 或 ZNSSD，默认 ZNSSD）
  光度 Adam 优化（见 `pin_solver.md`）。
- ZNSSD 的窗口统计（`kernel_size=7`、`epsilon=1e-6`）在 `znssd.cpp` 实现，
  逐窗口归一化；`compute_windows` 提供显式窗口版本。
- 实现与测试：`src/loss/*.cpp`，覆盖见 `tests/cpp`。
