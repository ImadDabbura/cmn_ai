# Welcome to `cmn_ai`

<p align="center">
<img src="https://raw.githubusercontent.com/ImadDabbura/cmn_ai/main/logo.png" width="300" height="200">
</p>

[![Python Versions](https://img.shields.io/badge/python-_3.13-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/cmn_ai.svg)](https://pypi.org/project/cmn_ai/)
[![PyPI downloads](https://img.shields.io/pypi/dm/cmn_ai.svg)](https://pypi.org/project/cmn_ai/)
[![codecov](https://codecov.io/gh/imaddabbura/cmn_ai/branch/main/graph/badge.svg)](https://codecov.io/gh/imaddabbura/cmn_ai)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

---

# `cmn_ai`

**A practical toolkit to accelerate your AI, Deep Learning, and Data Science workflows.**

`cmn_ai` is a Python library born from extensive, real-world experience in building and training machine learning models. It's designed to handle the most common and repetitive tasks in an ML/AI project, from data exploration to model deployment.

This library is built on the principle of **Boyd's Law**: _speed of iteration beats quality of iteration_. By providing robust, reusable components, `cmn_ai` helps you move faster, experiment more, and deliver results sooner.

## Key Features 🚀

- **Accelerated Workflow**: Stop rewriting boilerplate code. `cmn_ai` provides ready-to-use modules for data preprocessing, model development, and training.
- **Best Practices Baked-In**: Benefit from years of experience. The library incorporates proven software engineering and machine learning best practices so you can build high-quality models with confidence.
- **PyTorch & Scikit-Learn Integration**: The deep learning modules, including the flexible `Learner` and callbacks, are built for **PyTorch**. The tabular data tools are fully compatible with **scikit-learn's** `Pipeline` and `ColumnTransformer`.
- **Continuously Evolving**: The field of AI/ML moves fast, and so does `cmn_ai`. The library is actively maintained and updated with new functionalities.

---

## Installation

The easiest way to get started is to install `cmn_ai` directly from PyPI:

```sh
pip install cmn-ai
```

---

## Documentation

For a deep dive into the library's components and for full API details, please visit the official documentation:

**[https://imaddabbura.github.io/cmn_ai/](https://imaddabbura.github.io/cmn_ai/)**

---

## Development Setup

Interested in contributing or just want to explore the source code? The fastest way to set up a development environment is with **Poetry**.

1.  **Install Poetry:**
    ```sh
    pip install poetry
    ```
2.  **Clone the repository and install dependencies:**
    ```sh
    git clone https://github.com/ImadDabbura/cmn_ai
    cd cmn_ai
    poetry install
    ```

---

## Running Tests

To run the full test suite, use the following command:

```sh
poetry run pytest
```

---

## Contributing

`cmn_ai` is not currently open for contributions, as there is some foundational infrastructure work that needs to be completed first. However, it will be open to the community in the future\! Please check back for updates on how to get involved.

---

## License

`cmn_ai` is licensed under the Apache License, Version 2.0. You can find the full license text in the [LICENSE](LICENCE) file.
