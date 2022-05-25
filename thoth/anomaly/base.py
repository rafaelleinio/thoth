from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import List

from thoth.profiler import Metric, ProfilingReport


@dataclass
class Point:
    ts: datetime.datetime
    value: float


@dataclass
class TimeSeries:
    metric: Metric
    points: List[Point]

    def __lt__(self, other: TimeSeries) -> bool:
        return self.metric < other.metric


def convert_to_timeseries(history: List[ProfilingReport]) -> List[TimeSeries]:
    last_report = history[-1]
    metrics = [
        profiling_value.metric for profiling_value in last_report.profiling_values
    ]
    return sorted(
        [
            TimeSeries(
                metric=metric,
                points=[
                    Point(ts=report.ts, value=report.get_profiling_value(metric).value)
                    for report in history
                ],
            )
            for metric in metrics
        ]
    )
