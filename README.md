> ⚠️ **WIP** 👷 - soon to be released to [pypi.org](https://pypi.org)

____

![](https://i.imgur.com/UJwvBFC.png)

_data quality profiling monitoring tool._

![Python Version](https://img.shields.io/badge/python-3.9-brightgreen.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![flake8](https://img.shields.io/badge/code%20quality-flake8-blue)](https://github.com/PyCQA/flake8)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![pytest coverage: 100%](https://img.shields.io/badge/pytest%20coverage-100%25-green)](https://github.com/pytest-dev/pytest)

![NumPy](https://img.shields.io/badge/pyspark-%23FF6F00.svg?style=for-the-badge&logo=apachespark&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)


# Introduction
While the data and AI-driven culture emerge in several organizations, it is well
known that there are still many challenges in creating an efficient data operation. One
of the main barriers is **achieving high-quality data**. While more data brings more 
opportunities within the context of analytics and machine learning products, covering
this growing range of assets with quality checks becomes a real scalability issue. So
the big question is: how to create an efficient data quality service that covers as many
datasets as possible, does not require a lot of manual tuning, is computationally 
scalable, and with results that are easy to interpret?

This project main proposal is an automated end-to-end profiling-based data quality 
architecture. It implements profiling metrics computation, model optimization, anomaly 
detection, and generation of reports with high explainability.

By employing the most recent tools for data processing and AutoML aligned with modern 
data platform patterns it was possible to create an easy-to-use framework to empower 
developers and data users to build this solution.

## The Metrics Repository
![](media/arch.png)
The figure shows an overview of the entire flow: from the raw data to the 
decision-making regarding evaluating data quality.

First, in A, the raw dataset is transformed into aggregated profiling metrics by the 
profiler module and then saved in the Metrics Repository.

In B, all historical profiling from a given dataset is pulled
and used to optimize (train, evaluate, and select the best forecast model for each 
metric) and score all metrics. The anomaly scoring module implements this flow. The 
forecasts, scorings (errors), and optimizations for each metric are saved back to 
Metrics Repository.

Lastly, flow C, which is implemented by the quality assessment 
module, pulls the anomaly scorings for the latest data point and triggers a warning 
depending on the tolerance threshold found in the optimization, alerting the dataset 
owner about possible quality issues in the latest batch of data. 


## Monitor data profiling with simple commands! 🧐
```Python

import thoth as th

# init the Metrics Repository database
th.init_db(clear=True)

# profile the historical data, register the dataset in the Metrics Repository and 
# optimize ML models for all profiling time series.
th.profile_create_optimize(
    df=history_df,  # all your historical data
    dataset_uri="temperatures",  # identification for the dataset
    ts_column="ts",  # timestamp partition column
    session=session,  # sql session
    spark=spark,  # spark session
)

# assessing data quality for a new batch of data
th.assess_new_ts(
    df=new_batch_df,
    ts=datetime.datetime(1981, 12, 26),
    dataset_uri="temperatures",
    session=session
)


```

## Install the Thoth Python framework
```shell
pip install pythoth
```

## Quick Start in 2 simple steps

### 1) Start Dashboard and database (docker compose):

```shell
make app
```
Now the database for the Metrics Repository should be up and running, you can also 
access the dashboard at http://localhost:8543

![img.png](media/dashboard.png)

### 2) Test the framework with the example notebooks (docker compose)
```
make notebook-examples
```
You can open the notebook at http://localhost:8888

## Development
After creating your virtual environment:

#### Install dependencies

```bash
make requirements
```

#### Code Style and Quality
Apply code style (black and isort)
```bash
make apply-style
```

Run all checks (flake8 and mypy)
```
make checks
```

#### Testing and Coverage
```bash
make tests
```
