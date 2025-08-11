# cmn_ai

> A high-performance machine learning library for accelerating AI, Deep Learning, and Data Science workflows

<p align="center">
  <img src="https://raw.githubusercontent.com/ImadDabbura/cmn_ai/main/logo.png" width="300" height="200" alt="cmn_ai logo">
</p>

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/cmn_ai.svg)](https://pypi.org/project/cmn_ai/)
[![Downloads](https://img.shields.io/pypi/dm/cmn_ai.svg)](https://pypi.org/project/cmn_ai/)
[![Coverage](https://codecov.io/gh/imaddabbura/cmn_ai/branch/main/graph/badge.svg)](https://codecov.io/gh/imaddabbura/cmn_ai)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## Overview

**cmn_ai** is a comprehensive Python machine learning library designed to accelerate AI, Deep Learning, and Data Science workflows. Built from extensive real-world experience, it provides robust, reusable components for PyTorch-based deep learning and scikit-learn compatible tabular data processing.

The library follows **Boyd's Law** — _speed of iteration beats quality of iteration_ — enabling rapid experimentation and faster delivery of machine learning solutions.

## Key Features

### 🚀 **Accelerated Development**

- Pre-built modules eliminate boilerplate code
- Flexible callback system for training customization
- Seamless integration with existing workflows

### 🎯 **Best Practices Built-In**

- Years of ML engineering experience distilled into reusable components
- Robust error handling and memory management
- Consistent APIs across all modules

### 🔧 **Framework Integration**

- **Deep Learning**: Built on PyTorch with flexible `Learner` architecture
- **Tabular ML**: Full scikit-learn `Pipeline` and `ColumnTransformer` compatibility
- **Visualization**: Integrated plotting utilities for models and data

### 📊 **Domain-Specific Tools**

- **Vision**: Computer vision utilities with `VisionLearner` and batch visualization
- **Text**: NLP preprocessing and dataset handling with `TextList`
- **Tabular**: EDA tools and scikit-learn compatible transformers

## Installation

### From PyPI (Recommended)

```bash
pip install cmn-ai
```

### Development Installation

```bash
git clone https://github.com/ImadDabbura/cmn_ai.git
cd cmn_ai
pip install poetry
poetry install
```

## Quick Start

### Deep Learning with Learner

```python
from cmn_ai.learner import Learner
from cmn_ai.callbacks.training import DeviceCallBack, Recorder

# Create a learner with callbacks
learner = Learner(model, dls, loss_func, opt_func, callbacks=[Recorder("lr")])
learner.add_callback(DeviceCallBack("cuda:0"))

# Train your model
learner.fit(epochs=10, lr=1e-3)
```

### Vision Tasks

```python
from cmn_ai.vision import VisionLearner

# Vision-specific learner with built-in utilities
vision_learner = VisionLearner(model, dls, loss_func)
vision_learner.show_batch()  # Visualize training data
vision_learner.fit(epochs=20, lr=1e-4)
```

### Tabular Data Processing

```python
import pandas as pd
from cmn_ai.tabular.preprocessing import DateTransformer
from sklearn.pipeline import Pipeline

# Scikit-learn compatible preprocessing
x = pd.DataFrame(
    pd.date_range(start=pd.to_datetime("1/1/2018"), end=pd.to_datetime("1/08/2018"))
)
tfm = DateTransformer(drop=False)
tfm.fit_transform(X_train, y_train)
```

## Core Architecture

### Learner System

The `Learner` class provides a flexible foundation for training deep learning models with:

- Exception-based callback system for fine-grained training control
- Automatic mixed precision support
- Built-in logging and metrics tracking
- Memory optimization utilities

### Callback Framework

Fine-grained training control through exception-based callbacks:

- `CancelBatchException`: Skip current batch
- `CancelStepException`: Skip optimizer step
- `CancelBackwardException`: Skip backward pass
- `CancelEpochException`: Skip current epoch
- `CancelFitException`: Stop training entirely

### Modular Design

```
cmn_ai/
├── learner.py          # Core Learner class
├── callbacks/          # Training callbacks
├── vision/            # Computer vision utilities
├── text/              # NLP processing tools
├── tabular/           # Traditional ML tools
├── utils/             # Core utilities
├── plot.py            # Visualization tools
└── losses.py          # Custom loss functions
```

## Examples

### Training Loop Customization

```python
from functools import partial
from cmn_ai.callbacks.schedule import BatchScheduler
from cmn_ai.callbacks.training import MetricsCallback, ProgressCallback
from torcheval.metrics import MulticlassAccuracy
import torch.optim as opt

sched = partial(opt.lr_scheduler.OneCycleLR, max_lr=6e-2, total_steps=100)
learner = Learner(model, dls, loss_func, opt_func)
learner.add_callbacks(
    [
        ProgressCallback(),
        BatchScheduler(sched),
        MetricsCallback(accuracy=MulticlassAccuracy(nm_classes=10)),
    ]
)

learner.fit(epochs=50, lr=1e-3)
```

### Custom Callback Creation

```python
from cmn_ai.callbacks import Callback


class CustomCallback(Callback):
    def after_batch(self):
        if self.loss < self.threshold:
            print(f"Threshold reached at batch {self.batch}")
```

## Documentation

**📖 [Full Documentation](https://imaddabbura.github.io/cmn_ai/)**

**TODO**:

- [ ] [API Reference](https://imaddabbura.github.io/cmn_ai/api/)
- [ ] [Tutorial Notebooks](https://imaddabbura.github.io/cmn_ai/tutorials/)
- [ ] [Advanced Usage](https://imaddabbura.github.io/cmn_ai/advanced/)

## Development

### Setup Development Environment

```bash
git clone https://github.com/ImadDabbura/cmn_ai.git
cd cmn_ai
poetry install
```

### Run Tests

```bash
# Full test suite
poetry run pytest

# With coverage
poetry run pytest --cov=cmn_ai

# Specific test file
poetry run pytest tests/test_learner.py
```

### Code Quality

```bash
# pre-commit will take care of all things formatting, linting, syntax errors,
# tests, etc.
pre-commit run --all-files
```

### Build Documentation

```bash
mkdocs serve    # Local development server
mkdocs build    # Build documentation
```

## Requirements

- **Python**: 3.13+
- **Core Dependencies**: PyTorch, scikit-learn, NumPy, pandas
- **Optional**: matplotlib, seaborn (for plotting)

## Roadmap

- [ ] Distributed training support
- [ ] Additional vision architectures
- [ ] Advanced NLP utilities
- [ ] AutoML capabilities
- [ ] Model deployment tools

## 🙌 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have suggestions for adding or removing projects, please fork the repo, make your changes, and create a pull request. You can also simply open an issue with the tag "enhancement".

**Stay tuned for contribution guidelines!**

## License

Licensed under the [Apache License 2.0](https://github.com/ImadDabbura/cmn_ai/blob/main/LICENSE).

## Citation

If you use cmn_ai in your research, please cite:

```bibtex
@software{cmn_ai,
  title={cmn_ai: A Machine Learning Library for Accelerated AI Workflows},
  author={Imad Dabbura},
  url={https://github.com/ImadDabbura/cmn_ai},
  year={2024}
}
```

---

<div align="center">
  <strong>Built with ❤️ for the ML community</strong>
</div>
