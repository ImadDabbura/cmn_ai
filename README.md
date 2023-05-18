# Welcome to `cmn_ai`

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## Introduction

I am a big believer in **Boyd's Law**, which states that _speed of iteration beats quality of iteration_. Following this principle, I created `cmn_ai` to incorporate the most common tasks we encounter in ML/AI from data exploration to model development and training. Additionally, I baked into it some of the best practices I learned in my career so I don't have to repeat myself on every project I work on.

It is worth noting that the majority of the DL code such as [`Learner`](cmn_ai/learner.py) and callbacks assume that [`pytorch`](https://pytorch.org/) is used as the underlying DL library. Also, tabular data preprocessors/models are compatible with [`sklearn`](https://github.com/scikit-learn/scikit-learn) so they can be used easily with its `Pipeline` or `ColumnTransformer`.

Given the nature of the progress in ML/AI, I will always keep adding more functionalities as time passes.

## Contributing

`cmn_ai` is not open now for contribution because there are some infrastructure work I need to finish to make it ready for more contributors.

## License

`cmn_ai` has Apache License, as found in the [LICENCE](LICENSE) file.
