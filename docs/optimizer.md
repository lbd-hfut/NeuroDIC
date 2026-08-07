# Optimizer

`optimizer/` 模块封装项目级神经优化循环（LibTorch autograd 上的迭代器），
与可微损失/模型配合使用。

## 文件与职责

```text
optimizer.hpp       Optimizer 接口：LossClosure = std::function<torch::Tensor()>
                    OptimizationResult {iterations, final_loss}
                    minimize(iterations, closure) / step()
adam.hpp            AdamOptimizer：LibTorch torch::optim::Adam 封装
                    AdamOptimizer(parameters, learning_rate)
lbfgs.hpp           LBFGSOptimizer：L-BFGS 封装外壳（未来集成）
convergence.hpp     ConvergenceMonitor {max_iterations, tolerance, converged(loss)}
                    —— 决定优化何时停止（当前为外壳）
scheduler.hpp       Scheduler：学习率调度外壳（step()）
```

## 使用边界

- **PIN 求解路径**：`PINSolver` 先做 seed MSE 预训练，再做 SSD/ZNSSD 光度
  Adam 优化（`src/solver/pin_solver.cpp`）。
- **NDeF 求解路径**：`NDeFSolver` / `NDeFSurfaceSolver` 使用 Adam 及
  auto-batch 调度（稠密细化阶段与 Python NDeF-DIC scheduler 对齐）。
- 收敛判定当前由 `ConvergenceMonitor` 外壳占位；实际停止条件由各求解器循环
  依据迭代数与损失演化实现。
- 优化器只操作 `torch::Tensor` 参数，全程保留 autograd。
