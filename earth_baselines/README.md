# EARTH Baselines

This directory contains the non-EARTH baseline models extracted from
`C:\Users\ASUS\Desktop\大学\大四\毕设\EARTH-master\src`.

Included models:
- `AR`
- `VAR`
- `cola_gnn`
- `STGCN`

Excluded on purpose:
- `earth_epi`
- CDE / ODE / policy / physics-specific modules

The code keeps the original constructor style from EARTH:
- models accept `args`
- graph models expect `data.m`, `data.d`, `data.adj`, and `data.orig_adj`
- time-series baselines expect `args.window` and `args.horizon`

Example import:

```python
from earth_baselines import AR, VAR, cola_gnn, STGCN
```
