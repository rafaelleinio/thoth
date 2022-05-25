from __future__ import annotations

import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from faker import Faker
from faker.providers import lorem


class Trend:
    def __init__(self, base_value: Optional[float] = 0, slope: float = 0):
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


class Noise:
    def __init__(
        self,
        var: float = 0,
        mean: float = 1,
        seed: Optional[int] = None,
        generator: RandomGenerator = None,
    ):
        self.generator = generator or NormalPercentageDeviation(var=var, seed=seed)
        self.mean = mean
        self.var = var

    def generate(self) -> float:
        return self.generator.generate()


@dataclass(frozen=True)
class TimeSeries:
    date_points: np.ndarray
    value_points: np.ndarray


class TimeSeriesGenerator:
    def __init__(
        self,
        trend: Optional[Trend] = None,
        seasonality: Optional[Seasonality] = None,
        noise: Optional[Noise] = None,
    ):
        self.trend = trend or Trend()
        self.seasonality = seasonality or Seasonality()
        self.noise = noise or Noise()

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


class BatchDatasetGenerator:
    def __init__(
        self,
        events_generator: TimeSensitiveNumericFeatureGenerator,
        features: List[FeatureGenerator],
    ):
        self.events_generator = events_generator
        self.features = features

    def _generate_bath(self, time_context: TimeContext) -> pd.DataFrame:
        events_number = self.events_generator.generate(time_context=time_context)
        records = [
            {feature.name: feature.generate(time_context) for feature in self.features}
            for _ in range(int(events_number))
        ]
        return pd.DataFrame(records)

    def _cast_types(self, pdf: pd.DataFrame) -> pd.DataFrame:
        for feature in self.features:
            pdf = feature.cast(pdf)
        return pdf

    def generate(self, start_ts: datetime, n: int) -> pd.DataFrame:
        time_points = [
            TimeContext(start_ts=start_ts, ts=ts)
            for ts in pd.date_range(start_ts, periods=n).tolist()
        ]
        batches = [
            self._generate_bath(
                time_context=time_context,
            )
            for time_context in time_points
        ]
        return self._cast_types(pd.concat(batches))


class FeatureGenerator(ABC):
    def __init__(self, name: str, nulls_proportion: Optional[float] = 0.0):
        self.name = name
        self.nulls_proportion = nulls_proportion

    @abstractmethod
    def _generate(self, time_context: TimeContext) -> Any:
        pass

    def generate(self, time_context: TimeContext) -> Any:
        if (secrets.randbelow(10000) / 10000) < self.nulls_proportion:
            return None
        return self._generate(time_context)

    @abstractmethod
    def cast(self, pdf: pd.DataFrame) -> pd.DataFrame:
        """."""


class IdFeatureGenerator(FeatureGenerator):
    def __init__(
        self,
        name: str,
        min_id: int = 1,
        max_id: int = 1_000_000_000,
        monotonically_increase: bool = False,
        nulls_proportion: Optional[float] = 0.0,
    ):
        super().__init__(name, nulls_proportion)
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

    def cast(self, pdf: pd.DataFrame) -> pd.DataFrame:
        return pdf.astype({self.name: "int64"})


class TimestampFeatureGenerator(FeatureGenerator):
    def __init__(
        self,
        name: str,
        nulls_proportion: Optional[float] = 0.0,
    ):
        super().__init__(name, nulls_proportion)

    def _generate(self, time_context: TimeContext) -> datetime:
        base_date = datetime(
            year=time_context.ts.year,
            month=time_context.ts.month,
            day=time_context.ts.day,
            tzinfo=timezone.utc,
        )
        rand_microsecond = secrets.randbelow(86400)
        return base_date + timedelta(seconds=rand_microsecond)

    def cast(self, pdf: pd.DataFrame) -> pd.DataFrame:
        pdf[self.name] = pdf[self.name].astype("datetime64[ns, utc]")
        return pdf


class NumericFeatureGenerator(FeatureGenerator):
    def __init__(
        self,
        name: str,
        base_value: float,
        percentage_deviation_generator: Optional[RandomGenerator] = None,
        nulls_proportion: Optional[float] = 0.0,
    ):
        super().__init__(name, nulls_proportion)
        self.base_value = base_value
        self.percentage_deviation_generator = (
            percentage_deviation_generator or NormalPercentageDeviation()
        )

    def _generate(self, time_context: TimeContext) -> float:
        return self.base_value * self.percentage_deviation_generator.generate()

    def cast(self, pdf: pd.DataFrame) -> pd.DataFrame:
        return pdf.astype({self.name: "float64"})


class TimeSensitiveNumericFeatureGenerator(FeatureGenerator):
    def __init__(
        self,
        name: str,
        trend: Optional[Trend],
        seasonality: Optional[Seasonality] = None,
        noise: Optional[Noise] = None,
        nulls_proportion: Optional[float] = 0.0,
    ):
        super().__init__(name, nulls_proportion)
        self.trend = trend
        self.seasonality = seasonality or Seasonality()
        self.noise = noise or Noise()

    def _generate(self, time_context: TimeContext) -> float:
        trend_base_value = self.trend.get_value(time_context.start_ts, time_context.ts)
        seasonality_value = self.seasonality.get_constant(time_context.ts)
        noise_value = self.noise.generate()
        return trend_base_value * seasonality_value * noise_value

    def cast(self, pdf: pd.DataFrame) -> pd.DataFrame:
        return pdf.astype({self.name: "float64"})


class TextFeatureGenerator(FeatureGenerator):
    def __init__(
        self,
        name: str,
        max_base_length: int,
        percentage_deviation_generator: Optional[RandomGenerator] = None,
    ):
        super().__init__(name)
        self.max_base_length = max_base_length
        self.percentage_deviation = (
            percentage_deviation_generator or NormalPercentageDeviation()
        )
        self.fake = Faker()
        self.fake.add_provider(lorem)

    def _generate(self, time_context: TimeContext) -> str:
        max_length = self.max_base_length * self.percentage_deviation.generate()
        return self.fake.text(max_nb_chars=max_length)

    def cast(self, pdf: pd.DataFrame) -> pd.DataFrame:
        return pdf.astype({self.name: "string"})


class CategoryFeatureGenerator(FeatureGenerator):
    def __init__(self, name: str, categories: List[str]):
        super().__init__(name)
        self.categories = categories

    def _generate(self, time_context: TimeContext) -> str:
        return secrets.choice(self.categories)

    def cast(self, pdf: pd.DataFrame) -> pd.DataFrame:
        return pdf.astype({self.name: "string"})
