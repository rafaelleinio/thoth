import abc
import datetime
from typing import List, Optional, Tuple

from sqlmodel import Session, select
from sqlmodel.sql.expression import Select, SelectOfScalar

from thoth.anomaly import AnomalyScoring
from thoth.anomaly.optimization import AnomalyOptimization
from thoth.dataset import Dataset
from thoth.profiler import ProfilingReport

# supress warnings 😔 from issue: https://github.com/tiangolo/sqlmodel/issues/189
SelectOfScalar.inherit_cache = True
Select.inherit_cache = True


class MetricsRepositoryError(Exception):
    """Generic error for the metrics repository adapter operations."""


def _normalize_start_end_ts(
    start_ts: Optional[datetime.datetime], end_ts: Optional[datetime.datetime]
) -> Tuple[datetime.datetime, datetime.datetime]:
    return start_ts or datetime.datetime.min, end_ts or datetime.datetime.max


def _validate_profiling_records(
    dataset: Dataset, profiling_records: List[ProfilingReport]
) -> None:
    profiling_records_metrics = set(
        sum([list(pr.get_metrics()) for pr in profiling_records], [])
    )
    if list(profiling_records_metrics) != dataset.metrics:
        raise MetricsRepositoryError(
            "Given profiling have different metrics than the dataset."
            f"\nProfiling metrics={profiling_records_metrics}."
            f"\nDataset metrics={dataset.metrics}."
        )

    [profiling_records_dataset_uri] = set([pr.dataset_uri for pr in profiling_records])
    if profiling_records_dataset_uri != dataset.uri:
        raise MetricsRepositoryError(
            "Given profiling have different dataset_uri than the dataset."
            f"\nProfiling uri={profiling_records_dataset_uri}."
            f"\nDataset uri={dataset.uri}."
        )

    [profiling_records_granularity] = set([pr.granularity for pr in profiling_records])
    if profiling_records_granularity != dataset.granularity:
        raise MetricsRepositoryError(
            "Given profiling have different granularity than the dataset."
            f"\nProfiling uri={profiling_records_granularity}."
            f"\nDataset uri={dataset.granularity}."
        )


class AbstractRepository(abc.ABC):
    """IO operations for the Metrics Repository."""

    def add_profiling(self, dataset_uri: str, records: List[ProfilingReport]) -> None:
        """Add profiling records to the metrics repository.

        This operation implements an upsert logic. There should not to be duplicated
        profiling ids on the repository.

        Args:
            dataset_uri: dataset uri to search in the repository.
            records: collection of profiling records to add to the repository.

        """
        dataset = self.get_dataset(dataset_uri=dataset_uri)
        if not dataset:
            raise MetricsRepositoryError(f"Dataset with uri={dataset_uri} not found.")
        _validate_profiling_records(dataset=dataset, profiling_records=records)
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
        dataset_uri: str,
        start_ts: Optional[datetime.datetime] = None,
        end_ts: Optional[datetime.datetime] = None,
    ) -> List[ProfilingReport]:
        """Query the repository for profiling records for a given dataset and ts range.

        Args:
            dataset_uri: dataset URI to search for.
            start_ts: minimum timestamp to filter (closed interval).
            end_ts: maximum timestamp to filter (closed interval).

        Returns:
            Collection of profiling records found in the repository.

        """
        start_ts, end_ts = _normalize_start_end_ts(start_ts, end_ts)
        return self._select_profiling(
            dataset_uri=dataset_uri, start_ts=start_ts, end_ts=end_ts
        )

    @abc.abstractmethod
    def _select_profiling(
        self, dataset_uri: str, start_ts: datetime.datetime, end_ts: datetime.datetime
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
        dataset_uri: str,
        start_ts: Optional[datetime.datetime] = None,
        end_ts: Optional[datetime.datetime] = None,
    ) -> List[AnomalyScoring]:
        """Query the repository for scoring events for a given dataset and ts range.

        Args:
            dataset_uri: dataset URI to search for.
            start_ts: minimum timestamp to filter (closed interval).
            end_ts: maximum timestamp to filter (open interval).

        Returns:
            Collection of scoring events found in the repository.

        """
        start_ts, end_ts = _normalize_start_end_ts(start_ts, end_ts)
        return self._select_scoring(
            dataset_uri=dataset_uri, start_ts=start_ts, end_ts=end_ts
        )

    @abc.abstractmethod
    def _select_scoring(
        self, dataset_uri: str, start_ts: datetime.datetime, end_ts: datetime.datetime
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

    def get_optimization(self, dataset_uri: str) -> Optional[AnomalyOptimization]:
        """Retrieve from the repository the optimization for a specific dataset.

        Args:
            dataset_uri: dataset URI to search for.

        Returns:
            optimization object if dataset exist, None otherwise.

        """
        return self._get_optimization(dataset_uri=dataset_uri)

    @abc.abstractmethod
    def _get_optimization(self, dataset_uri: str) -> Optional[AnomalyOptimization]:
        """Child class must implement this operation."""

    def get_dataset(self, dataset_uri: str) -> Optional[Dataset]:
        """Find and return a specific dataset metadata from the repository.

        Args:
            dataset_uri: dataset uri.

        Returns:
            Dataset if the uri is found in the repository.

        """
        return self._get_dataset(dataset_uri=dataset_uri)

    @abc.abstractmethod
    def _get_dataset(self, dataset_uri: str) -> Optional[Dataset]:
        """Child class must implement this operation."""

    def get_datasets(self) -> List[Dataset]:
        """Get all datasets registered in the Metrics Repository."""
        return self._get_datasets()

    @abc.abstractmethod
    def _get_datasets(self) -> List[Dataset]:
        """Child class must implement this operation."""

    def add_dataset(self, dataset: Dataset) -> None:
        """Register a new dataset in the Metrics Repository.

        Args:
            dataset: dataset metadata.

        """
        return self._add_dataset(dataset=dataset)

    @abc.abstractmethod
    def _add_dataset(self, dataset: Dataset) -> None:
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
        new_record = self.get_scoring(id_=scoring.id_) or scoring
        for key, value in scoring.dict().items():
            setattr(new_record, key, value)
        self.session.add(new_record)
        self.session.commit()
        self.session.refresh(scoring)

    def _add_optimization(self, optimization: AnomalyOptimization) -> None:
        new_record = (
            self.get_optimization(dataset_uri=optimization.dataset_uri) or optimization
        )
        for key, value in optimization.dict().items():
            setattr(new_record, key, value)
        self.session.add(new_record)
        self.session.commit()
        self.session.refresh(optimization)

    def _select_profiling(
        self, dataset_uri: str, start_ts: datetime.datetime, end_ts: datetime.datetime
    ) -> List[ProfilingReport]:
        return self.session.exec(
            select(ProfilingReport).where(
                ProfilingReport.dataset_uri == dataset_uri
                and start_ts <= ProfilingReport.ts <= end_ts
            )
        ).all()

    def _get_profiling(self, id_: str) -> Optional[ProfilingReport]:
        return self.session.exec(
            select(ProfilingReport).where(ProfilingReport.id_profiling == id_)
        ).first()

    def _select_scoring(
        self, dataset_uri: str, start_ts: datetime.datetime, end_ts: datetime.datetime
    ) -> List[AnomalyScoring]:
        return self.session.exec(
            select(AnomalyScoring).where(
                AnomalyScoring.dataset_uri == dataset_uri
                and start_ts <= AnomalyScoring.ts <= end_ts
            )
        ).all()

    def _get_scoring(self, id_: str) -> Optional[AnomalyScoring]:
        return self.session.exec(
            select(AnomalyScoring).where(AnomalyScoring.id_ == id_)
        ).first()

    def _get_optimization(self, dataset_uri: str) -> Optional[AnomalyOptimization]:
        return self.session.exec(
            select(AnomalyOptimization).where(
                AnomalyOptimization.dataset_uri == dataset_uri
            )
        ).first()

    def _get_dataset(self, dataset_uri: str) -> Optional[Dataset]:
        return self.session.exec(
            select(Dataset).where(Dataset.uri == dataset_uri)
        ).first()

    def _get_datasets(self) -> List[Dataset]:
        return self.session.exec(select(Dataset)).all()

    def _add_dataset(self, dataset: Dataset) -> None:
        new_record = self.get_dataset(dataset_uri=dataset.uri) or dataset
        for key, value in dataset.dict().items():
            setattr(new_record, key, value)
        self.session.add(new_record)
        self.session.commit()
        self.session.refresh(dataset)
