import abc
import datetime
from typing import List, Optional, Tuple

from sqlmodel import Session, select
from sqlmodel.sql.expression import Select, SelectOfScalar

from thoth.anomaly import AnomalyScoring
from thoth.anomaly.optimization import AnomalyOptimization
from thoth.profiler import ProfilingReport

# supress warnings 😔 from issue: https://github.com/tiangolo/sqlmodel/issues/189
SelectOfScalar.inherit_cache = True
Select.inherit_cache = True


# SQLModelType = TypeVar("SQLModelType", bound=SQLModel)


def _normalize_start_end_ts(
    start_ts: Optional[datetime.datetime], end_ts: Optional[datetime.datetime]
) -> Tuple[datetime.datetime, datetime.datetime]:
    return start_ts or datetime.datetime.min, end_ts or datetime.datetime.max


class AbstractRepository(abc.ABC):
    """IO operations for the Metrics Repository."""

    def add_profiling(self, records: List[ProfilingReport]) -> None:
        """Add profiling records to the metrics repository.

        This operation implements an upsert logic. There should not to be duplicated
        profiling ids on the repository.

        Args:
            records: collection of profiling records to add to the repository.

        """
        self._add_profiling(records=records)

    @abc.abstractmethod
    def _add_profiling(self, records: List[ProfilingReport]) -> None:
        """Child class must implement this operation."""

    def add_scoring(self, scoring: AnomalyScoring) -> None:
        """Add a scoring event to the metrics repository.

        This operation implements an upsert logic. There should not to be duplicated
        scoring ids on the repository.

        Args:
            scoring: scoring event to add to the repository.

        """
        self._add_scoring(scoring=scoring)

    @abc.abstractmethod
    def _add_scoring(self, scoring: AnomalyScoring) -> None:
        """Child class must implement this operation."""

    def add_optimization(self, optimization: AnomalyOptimization) -> None:
        """Add an optimization event to the metrics repository.

        This operation implements an upsert logic. There should not more than one
        optimization for the same dataset.

        Args:
            optimization: optimization event to add to the repository.

        """
        self._add_optimization(optimization=optimization)

    @abc.abstractmethod
    def _add_optimization(self, optimization: AnomalyOptimization) -> None:
        """Child class must implement this operation."""

    def select_profiling(
        self,
        dataset: str,
        start_ts: Optional[datetime.datetime] = None,
        end_ts: Optional[datetime.datetime] = None,
    ) -> List[ProfilingReport]:
        """Query the repository for profiling records for a given dataset and ts range.

        Args:
            dataset: name of the dataset to search.
            start_ts: minimum timestamp to filter (closed interval).
            end_ts: maximum timestamp to filter (closed interval).

        Returns:
            Collection of profiling records found in the repository.

        """
        start_ts, end_ts = _normalize_start_end_ts(start_ts, end_ts)
        return self._select_profiling(dataset=dataset, start_ts=start_ts, end_ts=end_ts)

    @abc.abstractmethod
    def _select_profiling(
        self, dataset: str, start_ts: datetime.datetime, end_ts: datetime.datetime
    ) -> List[ProfilingReport]:
        """Child class must implement this operation."""

    def get_profiling(self, id_: str) -> Optional[ProfilingReport]:
        """Retrieve from the repository a specific profiling record.

        Args:
            id_: unique identifier for the profiling.

        Returns:
            profiling object if id exist, None otherwise.

        """
        return self._get_profiling(id_=id_)

    @abc.abstractmethod
    def _get_profiling(self, id_: str) -> Optional[ProfilingReport]:
        """Child class must implement this operation."""

    def select_scoring(
        self,
        dataset: str,
        start_ts: Optional[datetime.datetime] = None,
        end_ts: Optional[datetime.datetime] = None,
    ) -> List[AnomalyScoring]:
        """Query the repository for scoring events for a given dataset and ts range.

        Args:
            dataset: name of the dataset to search.
            start_ts: minimum timestamp to filter (closed interval).
            end_ts: maximum timestamp to filter (open interval).

        Returns:
            Collection of scoring events found in the repository.

        """
        start_ts, end_ts = _normalize_start_end_ts(start_ts, end_ts)
        return self._select_scoring(dataset=dataset, start_ts=start_ts, end_ts=end_ts)

    @abc.abstractmethod
    def _select_scoring(
        self, dataset: str, start_ts: datetime.datetime, end_ts: datetime.datetime
    ) -> List[AnomalyScoring]:
        """Child class must implement this operation."""

    def get_scoring(self, id_: str) -> Optional[AnomalyScoring]:
        """Retrieve from the repository a specific scoring event.

        Args:
            id_: unique identifier for the scoring.

        Returns:
            scoring object if id exist, None otherwise.

        """
        return self._get_scoring(id_=id_)

    @abc.abstractmethod
    def _get_scoring(self, id_: str) -> Optional[AnomalyScoring]:
        """Child class must implement this operation."""

    def get_optimization(self, dataset: str) -> Optional[AnomalyOptimization]:
        """Retrieve from the repository the optimization for a specific dataset.

        Args:
            dataset: name of the dataset to search.

        Returns:
            optimization object if dataset exist, None otherwise.

        """
        return self._get_optimization(dataset=dataset)

    @abc.abstractmethod
    def _get_optimization(self, dataset: str) -> Optional[AnomalyOptimization]:
        """Child class must implement this operation."""


class SqlRepository(AbstractRepository):
    """Repository adapter implementation for sql-based databases."""

    def __init__(self, session: Session):
        self.session = session

    def _add_profiling(self, records: List[ProfilingReport]) -> None:
        for record in records:
            new_record = self.get_profiling(id_=record.id_profiling) or record
            for key, value in record.dict().items():
                setattr(new_record, key, value)
            self.session.add(new_record)
        self.session.commit()
        for record in records:
            self.session.refresh(record)

    def _add_scoring(self, scoring: AnomalyScoring) -> None:
        new_record = self.get_scoring(id_=scoring.id) or scoring
        for key, value in scoring.dict().items():
            setattr(new_record, key, value)
        self.session.add(new_record)
        self.session.commit()
        self.session.refresh(scoring)

    def _add_optimization(self, optimization: AnomalyOptimization) -> None:
        new_record = self.get_optimization(dataset=optimization.dataset) or optimization
        for key, value in optimization.dict().items():
            setattr(new_record, key, value)
        self.session.add(new_record)
        self.session.commit()
        self.session.refresh(optimization)

    def _select_profiling(
        self, dataset: str, start_ts: datetime.datetime, end_ts: datetime.datetime
    ) -> List[ProfilingReport]:
        return self.session.exec(
            select(ProfilingReport).where(
                ProfilingReport.dataset == dataset
                and start_ts <= ProfilingReport.ts <= end_ts
            )
        ).all()

    def _get_profiling(self, id_: str) -> Optional[ProfilingReport]:
        return self.session.exec(
            select(ProfilingReport).where(ProfilingReport.id_profiling == id_)
        ).first()

    def _select_scoring(
        self, dataset: str, start_ts: datetime.datetime, end_ts: datetime.datetime
    ) -> List[AnomalyScoring]:
        return self.session.exec(
            select(AnomalyScoring).where(
                AnomalyScoring.dataset == dataset
                and start_ts <= AnomalyScoring.ts <= end_ts
            )
        ).all()

    def _get_scoring(self, id_: str) -> Optional[AnomalyScoring]:
        return self.session.exec(
            select(AnomalyScoring).where(AnomalyScoring.id == id_)
        ).first()

    def _get_optimization(self, dataset: str) -> Optional[AnomalyOptimization]:
        return self.session.exec(
            select(AnomalyOptimization).where(AnomalyOptimization.dataset == dataset)
        ).first()
