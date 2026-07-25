# Geometry

Geometry has multiple engines under one module boundary.

Stereo geometry owns projection, triangulation, reference/current 3D
reconstruction, and calibrated 3D displacement.

NDeF geometry owns multi-view projection, reference surface projection, deformed
surface projection, visibility, and photometric sampling geometry.

If geometry is between model output and loss, it must use `torch::Tensor`
operations and preserve autograd.
