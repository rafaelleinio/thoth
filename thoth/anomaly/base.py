from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import List

from thoth.profiler import Metric, ProfilingReport


@dataclass
class _Point:
    ts: datetime.datetime
    value: float


@dataclass
class _TimeSeries:
    metric: Metric
    points: List[_Point]

    def __lt__(self, other: _TimeSeries) -> bool:
        return self.metric < other.metric


def _convert_to_timeseries(profiling: List[ProfilingReport]) -> List[_TimeSeries]:
    last_report = profiling[-1]
    metrics = [
        profiling_value.metric for profiling_value in last_report.profiling_values
    ]
    return sorted(
        [
            _TimeSeries(
                metric=metric,
                points=[
                    _Point(ts=report.ts, value=report.get_profiling_value(metric).value)
                    for report in profiling
                ],
            )
            for metric in metrics
        ]
    )
