from typing import Optional, Type

from sqlmodel import Session, SQLModel

from thoth import repository


def _build_repo(
    model_: Type[SQLModel],
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
) -> repository.AbstractRepository:
    if repo:
        return repo
    if session:
        return repository.SqlRepository(model=model_, session=session)
    raise ValueError("Both repo and session cannot be None, one must be set.")
