from setuptools import find_packages, setup

__package_name__ = "thoth"
__version__ = "0.1.0"
__repository_url__ = "https://github.com/rafaelleinio/jenny"

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name=__package_name__,
    version=__version__,
    url=__repository_url__,
    packages=find_packages(
        exclude=["tests", "pipenv", "env", "venv", "htmlcov", ".pytest_cache", "pip"]
    ),
    license="MIT",
    author="Rafael Leinio",
    description="Data quality and profiling tool powered by Apache Spark.",
    keywords="apache-spark pyspark data quality profiling",
    install_requires=requirements,
)
