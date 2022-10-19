import os
from io import open
from typing import Dict

from setuptools import find_packages, setup

about: Dict[str, str] = {}
with open(
    file=os.path.join("strider_challenge", "__metadata__.py"),
    mode="r",
    encoding="utf-8",
) as f:
    exec(f.read(), about)

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__description__"],
    author=about["__author__"],
    license=about["__license__"],
    url=about["__url__"],
    packages=find_packages(
        exclude=["tests", "pipenv", "env", "venv", "htmlcov", ".pytest_cache", "pip"]
    ),
    long_description=long_description,
    python_requires=">=3.9, <4",
    install_requires=requirements,
)
