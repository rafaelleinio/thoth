## Jenny
Welcome to Jenny Project! Data quality checks and data profiling made easy.

![](https://i.pinimg.com/originals/bf/88/1b/bf881b6f1fdb18d0f34dd9c47f42a6f2.gif)

Python Example:

```python
from thoth import my_lib


def foo(arg):
    return my_lib.awesome_function(arg)  # change me
```

This library supports Python version 3.7+

## Installing

```bash
pip install thoth
```

Or after listing `jenny` in your
`requirements.txt` file:

```bash
pip install -r requirements.txt
```

This will expose `my_lib` under `jenny` module:

```python
from thoth import my_lib


def foo():
    bar = my_lib.cool_method()
```

If you want to experiment the latest features on development branch just run:
```bash
pip install git+https://github.com/rafaelleinio/jenny.git@dev
```
> keep in mind that the development branch may be not stable

## Development Environment

At the bare minimum you'll need the following for your development
environment:

1. [Python 3.7.2](http://www.python.org/)


It is strongly recommended to also install and use [pyenv](https://github.com/pyenv/pyenv) or [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html) to use the project locally.

## Getting started

#### 1. Clone the project:

```bash
    git clone git@github.com:rafaelleinio/thoth.git
    cd thoth
```

#### 2. Setup the python environment for the project:

For example using virtualenv, in the root of the repository run the following:
```bash
python3.7 -m virtualenv venv
source venv/bin/activate
```

If you need to configure your development environment in your IDE, notice
that virtualenv Python will be under:
`/path/to/jenny/venv/bin/python`

##### Errors

If you receive one error of missing OpenSSL to run the `pyenv install`, you can try to fix running:

```bash
sudo apt install -y libssl1.0-dev
```

#### 3. Install dependencies

```bash
make requirements
```

##### Errors

If you receive one error like this one:
```bash
 "import setuptools, tokenize;__file__='/tmp/pip-build-98gth33d/googleapis-common-protos/setup.py';
 .... 
 failed with error code 1 in /tmp/pip-build-98gth33d/googleapis-common-protos/
```
 
You can try to fix running:

```bash
python -m pip install --upgrade pip setuptools wheel
```

## Development

### Tests

Just run `make tests` to check if your code is fine.

Unit tests rely under the [test module](https://github.com/rafaelleinio/jenny/tree/master/tests/unit)
and integration tests, under the [integration_test module](https://github.com/rafaelleinio/jenny/tree/master/tests/integration).

Run only unit tests:
`make unit-tests`

Run only integration tests:
`make integration-tests`

[pytest](https://docs.pytest.org/en/latest/)
is used to write all of this project's tests.

### Code Style, PEP8 & Formatting

Just run `make black` before you commit to format all code.

Check if everything is fine with `make flake8`.

This project follows the [Black Code Style](https://github.com/ambv/black)
which follows PEP8 and unifies style across the project's codebase.

Additionally [Flake 8](http://flake8.pycqa.org/en/latest/) is used to
check for other things such as unnecessary imports and code-complexity.

You can check Flake 8 and Black by running the following within the project root:
```bash
make checks
```

#### Important:
The build of your branch will not be accepted in Drone.io if the code is not in
conformity with this style. So please, check before open a PR.

## Release
TBD

## Contributing
Any contributions are welcome! Feel free to open Pull Requests.
Contributing guidelines are under development.
