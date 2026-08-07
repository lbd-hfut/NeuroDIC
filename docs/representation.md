# Representation

`representation/` 模块把神经模型原始输出解码为物理场（位移/视差/3D 变形/参考
表面）。解码位于模型输出与损失之间，**必须保留 autograd**。

## 文件与职责

```text
representation.hpp         接口 FieldRepresentation::decode(coordinates, model_output)
pin_displacement_field.hpp PINDisplacementField：平面 2D PIN 位移场
                           PINDisplacementFieldOptions（构造带选项）
pin_disparity_field.hpp    PINDisparityField：立体 PIN 图像空间对应（视差）
ndef_deformation_field.hpp NDeFDeformationField：NDeF 3D 变形场解码
ndef_surface_field.hpp     NDeFSurfaceField：NDeF 参考表面场解码
```

## 可微性契约

- `decode` 对坐标与模型输出可微；调用方（solver 的光度损失构造）不得在解码与
  损失之间打断 autograd 图。
- 实现默认不持有张量（`Ownership: implementations own no tensors`）。
- 坐标约定与输出通道契约仍在收尾（头文件 TODO），各场解码保持与对应 problem/
  solver 的约定一致：

  - `PINDisplacementField` → PIN 2D 位移（u, v）
  - `PINDisparityField` → 立体视差/对应
  - `NDeFDeformationField` → 3D 变形（NDeFSolver）
  - `NDeFSurfaceField` → 参考表面（多视角 NDeF）
