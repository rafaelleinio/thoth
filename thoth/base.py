from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import List

from thoth.profiler import Metric, ProfilingReport


@dataclass
class Point:
    """Point object which holds a value and a timestamp reference."""

    ts: datetime.datetime
    value: float


@dataclass
class TimeSeries:
    """Series of points for a specific metric."""

    metric: Metric
    points: List[Point]

    def __lt__(self, other: TimeSeries) -> bool:
        return self.metric < other.metric


def convert_to_timeseries(profiling: List[ProfilingReport]) -> List[TimeSeries]:
    """Transform a list of profiling reports to a list of profiling metric ts."""
    last_report = profiling[-1]
    metrics = [
        profiling_value.metric for profiling_value in last_report.profiling_values
    ]
    return sorted(
        [
            TimeSeries(
                metric=metric,
                points=[
                    Point(ts=report.ts, value=report.get_profiling_value(metric).value)
                    for report in profiling
                ],
            )
            for metric in metrics
        ]
    )
