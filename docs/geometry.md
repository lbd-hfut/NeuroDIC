# Geometry

多个几何引擎共存于 `geometry/` 模块边界内。凡位于模型输出与损失之间的几何运算
必须用 `torch::Tensor` 实现并保留 autograd；求解后阶段的 CPU 重建（立体三角化）
不属于可微路径。

## 文件与职责

```text
geometry.hpp            基础接口 Geometry（几何引擎标记，被损失构造消费）
projection.hpp          可微针孔投影：
                        world_to_camera(points, R, t)
                        project_points_with_depth(points, K, ...) -> ProjectionResult {uv, depth}
                        MultiViewProjectionResult {uv[N,V,2], depth[N,V]}
triangulation.hpp       CPU float64 DLT 三角化（非可微）：
                        ReconstructionOptions {max_reprojection_error=2.0,
                                               require_positive_depth=true,
                                               undistort_iterations=12}
                        ReconstructionResult {points[N,3], valid[N], ...}
stereo_geometry.hpp     StereoGeometry：左右相机对的重建与校准 3D 位移
                        reconstruct_reference / reconstruct_current / displacement_3d
ndef_geometry.hpp       NDeFGeometry：多视角可微几何
                        project_reference_surface(surface)
                        project_deformed_surface(surface, deformation)
                        visibility(surface)
visibility.hpp          compute_visibility(surface)：多视角可见性掩码（可微）
coordinate_transform.hpp transform_coordinates(coordinates, transform)：
                        图像/归一化/相机/世界坐标空间变换
```

## 使用边界

- **NDeF 可微路径**：`NDeFGeometry` 的投影/可见性参与光度损失，必须在
  `torch::Tensor` 上逐点可微。
- **立体求解后重建**：`StereoGeometry` 与 `triangulation.hpp` 是 CPU float64
  后处理（参考/当前视差 → 3D 点 → 3D 位移），由 `PINStereoSolver` 在求解完成后
  调用。
- 坐标变换辅助既服务可微路径（NDeF 表面采样）也服务装配阶段。
