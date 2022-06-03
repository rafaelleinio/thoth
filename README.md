![](https://i.imgur.com/UJwvBFC.png)

_data quality profiling monitoring tool._

![Python Version](https://img.shields.io/badge/python-3.9-brightgreen.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![flake8](https://img.shields.io/badge/code%20quality-flake8-blue)](https://github.com/PyCQA/flake8)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![pytest coverage: 100%](https://img.shields.io/badge/pytest%20coverage-100%25-green)](https://github.com/pytest-dev/pytest)

## Monitor data profiling metrics in just two lines of code! 🧐
```Python
import thoth as th

reports = th.profile(
    df=df,
    dataset="my-dataset",
    ts_column="ts"
)

th.MetricsRepository().add(reports)

```

## Getting started

```shell
pip install pythoth
```


## Development
From repository root directory:

#### Install dependencies

```bash
make requirements
```

#### Code Style
Check code style:
```bash
make style-check
```
Apply code style with black and isort
```bash
make apply-style
```

Check code quality with flake8
```bash
make quality-check
```

Check typing with mypy
```
make type-check
```

Run all checks
```
make checks
```
#### Testing and Coverage
Unit tests:
```bash
make unit-tests
```
Integration tests:
```bash
make integration-tests
```
All tests:
```bash
make tests
```
