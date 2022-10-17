from __future__ import annotations

import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from faker import Faker
from faker.providers import lorem
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    Row,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


class Trend:
    def __init__(self, base_value: float = 0, slope: float = 0):
        self.slope = slope
        self.base_value = base_value

    def get_value(self, start_ts: datetime, ts: datetime) -> float:
        x = (ts - start_ts).days
        return self.base_value + self.slope * x


class Seasonality:
    def __init__(
        self,
        week_days: Optional[List[float]] = None,
        month_period: Optional[List[float]] = None,
        year_months: Optional[List[float]] = None,
    ):
        self.week_days = week_days or [1] * 7
        self.month_period = month_period or [1] * 3
        self.year_months = year_months or [1] * 12

    def get_week_days_constant(self, ts: datetime) -> float:
        return self.week_days[ts.weekday()]

    def get_month_period_constant(self, ts: datetime) -> float:
        if ts.day > 20:
            return self.month_period[2]
        if ts.day > 10:
            return self.month_period[1]
        return self.month_period[0]

    def get_year_months_constant(self, ts: datetime) -> float:
        return self.year_months[ts.month - 1]

    def get_constant(self, ts: datetime) -> float:
        return (
            self.get_week_days_constant(ts)
            * self.get_month_period_constant(ts)
            * self.get_year_months_constant(ts)
        )


class RandomGenerator(ABC):
    @abstractmethod
    def generate(self) -> float:
        pass


class NormalPercentageDeviation(RandomGenerator):
    def __init__(self, var: float = 0.05, seed: Optional[int] = None):
        self.mean = 1
        self.var = var
        self.generator = np.random.default_rng(seed=seed)

    def generate(self) -> float:
        return abs(self.generator.normal(loc=self.mean, scale=self.var))


@dataclass(frozen=True)
class TimeSeries:
    date_points: np.typing.NDArray[np.float64]
    value_points: np.typing.NDArray[np.float64]


class TimeSeriesGenerator:
    def __init__(
        self,
        trend: Optional[Trend] = None,
        seasonality: Optional[Seasonality] = None,
        noise: Optional[RandomGenerator] = None,
    ):
        self.trend = trend or Trend()
        self.seasonality = seasonality or Seasonality()
        self.noise = noise or NormalPercentageDeviation()

    def generate(self, start_ts: datetime, n: int) -> TimeSeries:
        date_points = pd.date_range(start_ts, periods=n).to_numpy()

        trend_points = np.fromiter(
            (self.trend.get_value(start_ts=start_ts, ts=ts) for ts in date_points),
            dtype=float,
        )
        seasonality_points = np.fromiter(
            (self.seasonality.get_constant(ts) for ts in date_points), dtype=float
        )
        noise_points = np.fromiter(
            (self.noise.generate() for _ in date_points), dtype=float
        )
        value_points = trend_points * seasonality_points * noise_points
        return TimeSeries(date_points=date_points, value_points=value_points)


@dataclass
class TimeContext:
    start_ts: datetime
    ts: datetime


class FeatureGenerator(ABC):
    def __init__(
        self,
        name: str,
        nulls_proportion: float = 0.0,
        nulls_proportion_noise: Optional[RandomGenerator] = None,
    ):
        self.name = name
        self.nulls_proportion = nulls_proportion
        self.nulls_proportion_noise = (
            nulls_proportion_noise or NormalPercentageDeviation()
        )

    @abstractmethod
    def _generate(self, time_context: TimeContext) -> Any:
        pass

    def generate_nulls_proportion(self) -> float:
        return self.nulls_proportion * self.nulls_proportion_noise.generate()

    def generate(
        self, time_context: TimeContext, nulls_proportion: float = None
    ) -> Any:
        if (secrets.randbelow(10000) / 10000) < (
            nulls_proportion or self.generate_nulls_proportion()
        ):
            return None
        return self._generate(time_context)

    @property
    @abstractmethod
    def pandas_field(self) -> Dict[str, str]:
        """."""

    @property
    @abstractmethod
    def spark_field(self) -> StructField:
        """."""


class IdFeatureGenerator(FeatureGenerator):
    def __init__(
        self,
        name: str,
        min_id: int = 1,
        max_id: int = 1_000_000_000,
        monotonically_increase: bool = False,
        nulls_proportion: float = 0.0,
        nulls_proportion_noise: Optional[RandomGenerator] = None,
    ):
        super().__init__(name, nulls_proportion, nulls_proportion_noise)
        self.max_id = max_id
        self.min_id = min_id
        self.monotonically_increase = monotonically_increase
        self.counter = 0

    def _generate(self, time_context: TimeContext) -> int:
        if self.monotonically_increase:
            id_ = self.min_id + self.counter
            self.counter += 1
            return id_
        return secrets.randbelow(self.max_id - self.min_id) + self.min_id

    @property
    def pandas_field(self) -> Dict[str, str]:
        return {self.name: "int64"}

    @property
    def spark_field(self) -> StructField:
        return StructField(name=self.name, dataType=IntegerType(), nullable=True)


class TimestampFeatureGenerator(FeatureGenerator):
    def __init__(
        self,
        name: str,
        nulls_proportion: float = 0.0,
        nulls_proportion_noise: Optional[RandomGenerator] = None,
    ):
        super().__init__(name, nulls_proportion, nulls_proportion_noise)

    def _generate(self, time_context: TimeContext) -> datetime:
        base_date = datetime(
            year=time_context.ts.year,
            month=time_context.ts.month,
            day=time_context.ts.day,
            tzinfo=timezone.utc,
        )
        rand_microsecond = secrets.randbelow(86400)
        return base_date + timedelta(seconds=rand_microsecond)

    @property
    def pandas_field(self) -> Dict[str, str]:
        return {self.name: "datetime64[ns, utc]"}

    @property
    def spark_field(self) -> StructField:
        return StructField(name=self.name, dataType=TimestampType(), nullable=True)


class NumericFeatureGenerator(FeatureGenerator):
    def __init__(
        self,
        name: str,
        base_value: float,
        noise: Optional[RandomGenerator] = None,
        nulls_proportion: float = 0.0,
        nulls_proportion_noise: Optional[RandomGenerator] = None,
    ):
        super().__init__(name, nulls_proportion, nulls_proportion_noise)
        self.base_value = base_value
        self.percentage_deviation_generator = noise or NormalPercentageDeviation()

    def _generate(self, time_context: TimeContext) -> float:
        return self.base_value * self.percentage_deviation_generator.generate()

    @property
    def pandas_field(self) -> Dict[str, str]:
        return {self.name: "float64"}

    @property
    def spark_field(self) -> StructField:
        return StructField(name=self.name, dataType=DoubleType(), nullable=True)


class TimeSensitiveNumericFeatureGenerator(FeatureGenerator):
    def __init__(
        self,
        name: str,
        trend: Trend,
        seasonality: Optional[Seasonality] = None,
        noise: Optional[RandomGenerator] = None,
        nulls_proportion: float = 0.0,
        nulls_proportion_noise: Optional[RandomGenerator] = None,
    ):
        super().__init__(name, nulls_proportion, nulls_proportion_noise)
        self.trend = trend
        self.seasonality = seasonality or Seasonality()
        self.noise = noise or NormalPercentageDeviation()

    def _generate(self, time_context: TimeContext) -> float:
        trend_base_value = self.trend.get_value(time_context.start_ts, time_context.ts)
        seasonality_value = self.seasonality.get_constant(time_context.ts)
        noise_value = self.noise.generate()
        return trend_base_value * seasonality_value * noise_value

    @property
    def pandas_field(self) -> Dict[str, str]:
        return {self.name: "float64"}

    @property
    def spark_field(self) -> StructField:
        return StructField(name=self.name, dataType=DoubleType(), nullable=True)


class TextFeatureGenerator(FeatureGenerator):
    def __init__(
        self,
        name: str,
        max_base_length: int,
        percentage_deviation_generator: Optional[RandomGenerator] = None,
        nulls_proportion: float = 0.0,
        nulls_proportion_noise: Optional[RandomGenerator] = None,
    ):
        super().__init__(name, nulls_proportion, nulls_proportion_noise)
        self.max_base_length = max_base_length
        self.percentage_deviation = (
            percentage_deviation_generator or NormalPercentageDeviation()
        )
        self.fake = Faker()
        self.fake.add_provider(lorem)

    def _generate(self, time_context: TimeContext) -> str:
        max_length = self.max_base_length * self.percentage_deviation.generate()
        return str(self.fake.text(max_nb_chars=max_length))

    @property
    def pandas_field(self) -> Dict[str, str]:
        return {self.name: "string"}

    @property
    def spark_field(self) -> StructField:
        return StructField(name=self.name, dataType=StringType(), nullable=True)


class CategoryFeatureGenerator(FeatureGenerator):
    def __init__(
        self,
        name: str,
        categories: List[str],
        nulls_proportion: float = 0.0,
        nulls_proportion_noise: Optional[RandomGenerator] = None,
    ):
        super().__init__(name, nulls_proportion, nulls_proportion_noise)
        self.categories = categories

    def _generate(self, time_context: TimeContext) -> str:
        return secrets.choice(self.categories)

    @property
    def pandas_field(self) -> Dict[str, str]:
        return {self.name: "string"}

    @property
    def spark_field(self) -> StructField:
        return StructField(name=self.name, dataType=StringType(), nullable=True)


class BatchDatasetGenerator:
    def __init__(
        self,
        events_generator: TimeSensitiveNumericFeatureGenerator,
        features: List[FeatureGenerator],
    ):
        self.events_generator = events_generator
        self.features = features

    def _generate_batch(self, time_context: TimeContext) -> List[Dict[str, Any]]:
        events_number = self.events_generator.generate(time_context=time_context)
        features_nulls_proportions = [
            fg.generate_nulls_proportion() for fg in self.features
        ]
        records = [
            {
                feature.name: feature.generate(
                    time_context, nulls_proportion=nulls_proportion
                )
                for feature, nulls_proportion in zip(
                    self.features, features_nulls_proportions
                )
            }
            for _ in range(int(events_number))
        ]
        return records

    def generate(self, start_ts: datetime, n: int) -> List[Dict[str, Any]]:
        time_points = [
            TimeContext(start_ts=start_ts, ts=ts)
            for ts in pd.date_range(start_ts, periods=n).tolist()
        ]
        batches: List[List[Dict[str, Any]]] = [
            self._generate_batch(
                time_context=time_context,
            )
            for time_context in time_points
        ]
        flatten: List[Dict[str, Any]] = sum(batches, [])
        return flatten

    def to_pandas_df(self, dataset: List[Dict[str, Any]]) -> pd.DataFrame:
        pdf = pd.DataFrame(data=dataset)
        return pdf.astype(
            dtype={
                k: v
                for d in [feature.pandas_field for feature in self.features]
                for k, v in d.items()
            }
        )

    def to_spark_df(
        self, dataset: List[Dict[str, Any]], spark: SparkSession
    ) -> DataFrame:
        return spark.createDataFrame(
            data=[Row(**r) for r in dataset],
            schema=StructType(
                fields=[feature.spark_field for feature in self.features]
            ),
        )
