## LogAvgExp - Pytorch

Implementation of <a href='https://arxiv.org/abs/2111.01742'>LogAvgExp</a> for Pytorch

## Install

```bash
$ pip install logavgexp-pytorch
```

## Usage

```python
import torch
from logavgexp_pytorch import logavgexp

x = torch.randn(1, 2048, 5)
y = logavgexp(x, dim = 1) # (1, 5)
```

## Citations

@misc{lowe2021logavgexp,
    title   = {LogAvgExp Provides a Principled and Performant Global Pooling Operator}, 
    author  = {Scott C. Lowe and Thomas Trappenberg and Sageev Oore},
    year    = {2021},
    eprint  = {2111.01742},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
