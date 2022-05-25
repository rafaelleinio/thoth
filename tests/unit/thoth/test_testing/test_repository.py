import datetime

from thoth.profiler import ProfilingReport
from thoth.repository import MetricsRepository


class TestMetricsRepository:
    def test_get_profiling_history(self):
        # arrange
        ts = datetime.datetime(2022, 4, 19)
        td = datetime.timedelta(1)
        reports = [
            ProfilingReport("uuid1", "ds", ts + td, "DAY", []),
            ProfilingReport("uuid2", "ds", ts + td * 2, "DAY", []),
            ProfilingReport("uuid3", "ds", ts + td * 3, "DAY", []),
            ProfilingReport("uuid4", "ds", ts + td * 4, "DAY", []),
        ]
        repo = MetricsRepository()

        # act
        repo.add_profiling_report(reports=reports)
        output_reports = repo.get_profiling_history(
            dataset_name="ds", min_ts=ts + td * 2, max_ts=ts + td * 3
        )

        # assert
        assert reports[1:3] == output_reports
