# PyTorch implementation of Adaptive Traffic Forecasting on Daily Basis: A Spatio-Temporal Context Learning Approach


This is a testing PyTorch version implementation of ASTCL in the following paper: **"Adaptive Traffic Forecasting on Daily Basis: A Spatio-Temporal Context Learning Approach"**.

## Installation
```
pip install -r requirements.txt
```

## Citation

This repository contains a reference implementation for learning purposes. If you find it useful for your research, please cite the corresponding paper:
```
@article{li2025adaptive,
  title={Adaptive Traffic Forecasting on Daily Basis: A Spatio-Temporal Context Learning Approach},
  author={Li, Xiaoyu and Zhang, Yitian and Long, Guodong and Hu, Yupeng and Lu, Wenpeng and Chen, Meng and Zhang, Chengqi and Gong, Yongshun},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2025},
  publisher={IEEE}
}
```

## Tensor contract and regression tests

`ASTCL.forward(input, adj)` expects `input` shaped `[B, N, T, C]` and
`adj` shaped `[B, N, N]`. RevIN normalizes the last feature axis using
statistics reduced over the node and time axes; convolutions, JointAttention,
JointConv and GCN use `[B, C, N, T]`. Spatial-temporal attention flattens
`[B, N, T, C_partial]` to `[B, N*T, C_partial]`.

The result is `[B, T]`: feature 0 of node 0, matching the single-target labels
produced by the existing data pipeline. Batch and time axes are retained even
when their size is one. The current architecture preserves time length, so
`t_history` must equal `t_pred`; unequal horizons now raise an explicit error.

`preprocess_dataset` selects feature 0 from raw data, so the supplied configs use
`num_features=1` and `channels` beginning with `1`. Custom multivariate inputs are
supported when both settings match the actual last input dimension. Old
checkpoints trained with 2/3 input channels cannot be loaded with the corrected
single-feature configs without retraining or a matching multivariate pipeline.
Even where weight shapes match, corrected attention ordering changes behavior;
this fix does not establish compatibility with previously reported metrics.

After installing the runtime dependencies, install pytest and run:

```bash
python -m pip install pytest
python -m pytest tests -q
```

The tests cover forward/backward propagation, singleton dimensions, attention
token order, JointConv/GCN/RevIN, every supplied configuration, generated raw data
through preprocessing and DataLoader, one training/validation epoch, checkpoint
reload and both real training and evaluation CLI entrypoints. No external traffic
dataset or pretrained weights are needed. Local verification used Python 3.12.14,
PyTorch 2.14.1+cpu and NumPy 2.3.5; the original pinned environment and real-data
forecasting accuracy were not evaluated. The synthetic adjacency fixture can emit
the existing zero-degree inverse warning in `utils.asym_adj`; its infinities are
already replaced with zeros by that function.
