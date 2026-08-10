# Python Bindings

编译扩展为 `neurodic._neurodic`（pybind11 模块 `bindings/python/module.cpp`，
按子模块绑定 `bind_core / bind_data / bind_interpolation / bind_initialization /
bind_calibration / bind_opencv_calibration / bind_problem / bind_geometry /
bind_solver / bind_result`）。

绑定只暴露经过验证的 C++ 接口，不得包含科学算法，也不自动暴露每个内部类。

## Python 包（`python/neurodic`）

薄封装层，装配输入并调用扩展；`_neurodic` 不可导入时给出清晰错误
（`native_available()` 判断扩展是否可用）。

```text
python/neurodic/__init__.py   包入口：导出 calibrate / calibration / models /
                             seeds / ndef_dic / ndef_sparse_precalculation /
                             pretrain_ndef_surface / pin_dic / pin_stereo_dic /
                             NDeFROIOptions / generate_ndef_roi / configure_runtime
api/calibrate.py             calibrate(mode, ...)：MONO/STEREO/COLMAP 标定入口
api/ndef_dic.py              ndef_dic() / ndef_sparse_precalculation()
api/ndef_surface.py          pretrain_ndef_surface()：参考表面/深度网络预训练
api/pin_dic.py               pin_dic()：平面 2D PIN-DIC
api/pin_stereo_dic.py        pin_stereo_dic()：立体 PIN-DIC（含稠密表面融合与可视化）
calibration.py               传统标定装配：stereo/multiview(colmap_like) 流程、
                             尺度恢复（棋盘格/元模型 Sim(3)）、结果序列化
models.py                    神经模型选择（PIN/NDeF）与 torch 模型包装
seeds.py                     种子生成（SIFT/网格/传统）的 Python 入口
ndef_roi.py                  NDeFROIOptions / generate_ndef_roi
runtime.py                   configure_runtime()：设备/精度/随机种子
visualization/               calibration.py / seeds.py：标定与种子可视化
```

配置入口使用 `config/*.yaml`（`config/pin_2d.yaml`、`config/pin_stereo.yaml`、
`config/ndef_multi.yaml`、`config/calibration.yaml` 等）。
