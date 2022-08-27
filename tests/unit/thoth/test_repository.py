from sqlmodel import Field, Session, SQLModel

from thoth import repository


class MockSqlModel(SQLModel, table=True):
    name: str = Field(primary_key=True)
    age: int


class TestSqlRepository:
    def test_add_and_get(self, session: Session):
        # arrange
        repo = repository.SqlRepository(session=session, model=MockSqlModel)
        input1 = MockSqlModel(name="1", age=18)
        input2 = MockSqlModel(name="2", age=18)
        input3 = MockSqlModel(name="1", age=19)

        # act
        repo.add([input1])
        repo.add([input2, input3])
        output1 = repo.get(reference="1")
        output2 = repo.get(reference="2")

        # assert
        assert output1 == input3
        assert output2 == input2
