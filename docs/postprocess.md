# Postprocess

`postprocess/` 模块在求解完成后派生物理量并清理结果（非可微后处理）。

## 文件与职责

```text
displacement.hpp    displacement_magnitude(displacement)：位移幅值
filtering.hpp       filter_result(values)：剔除无效/离群值
physical_scale.hpp  apply_physical_scale(values, scale)：图像/像素空间场
                    转换为物理单位
strain.hpp          compute_strain(displacement)：由位移场派生应变
```

## 使用边界

- 后处理只消费求解器输出的场张量，不参与训练/优化（非可微路径）。
- 立体路径的 3D 重建与 3D 位移属于 `geometry/`（`StereoGeometry`、
  `triangulation.hpp`）；本模块负责二维/三维物理量的标定与派生。
- 物理尺度转换通常依赖标定结果（mm/px 尺度），由问题装配阶段提供 scale 参数。
