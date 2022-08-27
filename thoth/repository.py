import abc
from typing import Optional, Sequence, Type

from sqlalchemy.inspection import inspect
from sqlmodel import Session, SQLModel, select
from sqlmodel.sql.expression import Select, SelectOfScalar

# supress warnings 😔 from issue: https://github.com/tiangolo/sqlmodel/issues/189
SelectOfScalar.inherit_cache = True  # type: ignore
Select.inherit_cache = True  # type: ignore


class AbstractRepository(abc.ABC):
    """Base abstract repository pattern adapter."""

    def __init__(self, model: Type[SQLModel]):
        self.model = model

    @abc.abstractmethod
    def _add(self, records: Sequence[SQLModel]) -> None:
        """Child class should implement add command logic."""

    def add(self, records: Sequence[SQLModel]) -> None:
        """Add records to the repository.
        Args:
            records: input records to add.
        """
        self._add(records)

    @abc.abstractmethod
    def _get(self, reference: str) -> Optional[SQLModel]:
        """Child class should implement get command logic."""

    def get(self, reference: str) -> Optional[SQLModel]:
        """Retrieve a specific record from repository.
        Args:
            reference: key to find the record.
        Returns:
            record for given key or None if not found.
        """
        return self._get(reference)


class SqlRepository(AbstractRepository):
    """Repository adapter implementation from sql-based databases."""

    def __init__(
        self, model: Type[SQLModel], session: Session, pk: Optional[str] = None
    ):
        super().__init__(model)
        self.session = session
        self.pk = pk or inspect(model).primary_key[0].name

    def _add(self, records: Sequence[SQLModel]) -> None:
        """This method implements a upsert logic for the add query.
        If the reference does not exist it creates for the first time. If already exist,
        it will update the features.
        """
        for record in records:
            new_record = self._get(reference=getattr(record, self.pk)) or record
            for key, value in record.dict().items():
                setattr(new_record, key, value)
            self.session.add(new_record)
        self.session.commit()

    def _get(self, reference: str) -> Optional[SQLModel]:
        statement = select(self.model).where(getattr(self.model, self.pk) == reference)
        return self.session.exec(statement).first()
