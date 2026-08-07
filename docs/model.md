# Model

`model/` 模块定义神经模型抽象与具体实现：PIN 分支（用户可选模型）与 NDeF 分支
（求解器内部控制拓扑）。

## 文件与职责

```text
neural_model.hpp        NeuralModel : torch::nn::Module，抽象 forward(coordinates)
fourier.hpp             FourierEncoding（PIN 与 NDeF 共享的固定 dyadic 位置编码）：
                        FourierEncodingOptions {enabled=true, num_frequencies=6,
                            include_input=true, angular_scale=π}
mlp.hpp                 MLPModel（PIN 分支，MSPINN 兼容 tanh MLP）：
                        PINModelOptions {input_dim=2, output_dim=2, hidden_dim=64,
                            hidden_layers=5, fourier_encoding}
ndef_internal_model.hpp  NDeFInternalModel（NDeF 3D 参考表面变形模型）：
                        NDeFModelOptions {hidden_dim=32, hidden_layers=5,
                            output_scale=1.0, fourier_encoding}
                        （构造带 coordinate_center / coordinate_scale 归一化）
ndef_depth_model.hpp     NDeFDepthModel（C++/LibTorch 版 SfMDepthFiLMNet）：
                        NDeFDepthModelOptions {hidden_dim=32, pixel_layers=3,
                            camera_layers=2, trunk_layers=3,
                            camera_embedding_dim=16, positional_encoding_*}
                        forward(normalized_uv, camera_indices)
model_factory.hpp       ModelFactory：按名称创建用户可选 PIN 模型
resnet.hpp / siren.hpp  ResNet / SIREN 模型外壳（未来 PIN 分支候选）
```

## 使用边界

- **PIN**：`ModelFactory` 创建用户可选的 `MLPModel`（tanh + 可选 Fourier）；
  Python `models.py` 暴露选择。
- **NDeF**：`NDeFInternalModel` 由 `NDeFSolver` 内部控制（不对外暴露层数/宽度/
  跳连）；`NDeFDepthModel` 由 `NDeFSurfaceSolver` 使用（参考表面/深度网络，
  `config/ndef_multiview.yaml` 配置预训练与稠密细化超参）。
- 模型输出是神经场原始输出，由 `representation/` 解码为物理场后再进入损失。
- 全部模型继承 `torch::nn::Module`，参数优化走 LibTorch autograd。
