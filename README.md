# CFGAN

This repository supports the development and validation of the paper: "CFGAN: Complex frequency-domain graph attention network for time series forecasting".

## Contents

- `cfgan.py`: model, signed correlation graphs, CSFEM, TDFFM, and the recurrent head
- `cfgan_dataloader.py`: daily stock batches and tensor loading
- `cfgan_utils.py`: losses and evaluation metrics
- `learn_cfgan.py`: training, evaluation, and inference
- `data/csi300_stock_index.npy`: instrument-to-integer-id mapping

Generated outputs are not tracked. This includes checkpoints, TensorBoard events, logs, and prediction files.

## Setup

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

CFGAN reads Alpha360 features through Qlib, so a local Qlib data directory is required:

```bash
# official collection script
python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

# or a community Qlib binary release
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1
rm -f qlib_bin.tar.gz
```

## Data

Market data are not included in this repository. `data/csi300_stock_index.npy` is only a mapping from instrument identifiers to integer ids and contains no prices.

CFGAN uses the six daily Alpha360 features (open, close, high, low, VWAP, volume) over the preceding 60 trading days, giving a 360-dimensional input vector per stock. The default split is training 2007-2017, validation 2018-2019, and test 2020-2023.

For CSI500, build the matching mapping file first:

```python
import os
import numpy as np
import qlib
from qlib.config import REG_CN
from qlib.data import D

qlib.init(provider_uri=os.path.expanduser("~/.qlib/qlib_data/cn_data"), region=REG_CN)
instruments = D.list_instruments(
    D.instruments("csi500"), start_time="2007-01-01", end_time="2023-12-31", as_list=True
)
stock_index = {symbol: i for i, symbol in enumerate(sorted(instruments))}
np.save("data/csi500_stock_index.npy", stock_index)
```

## Usage

```bash
python learn_cfgan.py \
  --provider_uri ~/.qlib/qlib_data/cn_data \
  --data_set csi300 \
  --stock_index ./data/csi300_stock_index.npy \
  --model_name CFGAN \
  --name csi300_CFGAN \
  --batch_size -1 \
  --n_epochs 200 \
  --repeat 1 \
  --seed 42
```

Keep `--batch_size -1`. Each batch then covers all stocks of a single trading day, which is what the correlation graph is built on. A positive batch size shuffles rows across dates and changes the meaning of that graph.

For a quick pipeline check, run a few trading days only:

```bash
python learn_cfgan.py \
  --provider_uri ~/.qlib/qlib_data/cn_data \
  --data_set csi300 --stock_index ./data/csi300_stock_index.npy \
  --model_name CFGAN --name csi300_smoke \
  --batch_size -1 --n_epochs 1 --repeat 1 --seed 42 \
  --train_start_date 2020-01-02 --train_end_date 2020-01-06 \
  --valid_start_date 2020-01-07 --valid_end_date 2020-01-09 \
  --test_start_date 2020-01-10 --test_end_date 2020-01-14
```

Outputs are written to `output/<dataset>/<model>/train_.../valid_.../test_.../`, containing `log/`, `mode_save/`, and TensorBoard event files:

```bash
tensorboard --logdir ./output
```

## Notes

- Baselines are imported from Qlib. SDGNN, RTGNN, and FourierGNN are not included, and neither are the ablation variant or the portfolio backtesting workflow.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{li2026cfgan,
  author    = {Li, Zhihao and Gao, Lele and Qiu, Wenlong and Xiao, Wenyun and Wang, Hongnian},
  title     = {{CFGAN}: Complex Frequency-Domain Graph Attention Network for Time Series Forecasting},
  booktitle = {IEEE International Conference on Acoustics, Speech, and Signal Processing},
  year      = {2026},
  pages     = {1231--1235}
}
```

## Acknowledgements

We acknowledge the authors of [HIST](https://github.com/Wentao-Xu/HIST) (Xu et al., 2021), whose training and evaluation code our pipeline is adapted from. This project builds on [Qlib](https://github.com/microsoft/qlib) and [PyTorch](https://pytorch.org/). `ALSTMModel` in `cfgan.py` is adapted from Qlib's `pytorch_alstm.py`.

## License

MIT
