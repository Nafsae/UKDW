# UKDW

UKDW is a medical image segmentation codebase built around a U-KAN backbone with wavelet enhancement and feature fusion.

## Repository Structure

```text
UKDW/
├── README.md
├── environment.yaml
├── train.py
├── train.sh
├── val.py
├── val.sh
├── datasets/
│   ├── __init__.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── kan.py
│   ├── layers.py
│   └── ukdw.py
└── utils/
    ├── __init__.py
    ├── losses.py
    ├── metrics.py
    └── misc.py
```

## Setup

```bash
conda env create -f environment.yaml
conda activate ukan
```

## Training

```bash
bash train.sh
```

## Validation

```bash
bash val.sh
```

## Main Model

```python
from models import UKDW
```
