from unittest.mock import Mock, patch

from thoth.logging import DEFAULT_LOGGING_SCHEMA, get_logger, set_logging_config


class Foo:
    def __init__(self):
        self.logger = get_logger()

    def foo(self):
        return self.logger


def foo():
    return get_logger()


@patch("logging.config.dictConfig")
def test_set_logging_config(mock_dict_config: Mock):
    # act
    set_logging_config()

    # assert
    mock_dict_config.assert_called_once_with(DEFAULT_LOGGING_SCHEMA)


def test_get_logger():
    # act
    logger_from_class = Foo().foo()
    logger_from_external_method = foo()
    logger_custom_name = get_logger("custom_name")

    # assert
    assert logger_from_class.name == "tests.unit.thoth.test_logging.Foo"
    assert logger_from_external_method.name == "tests.unit.thoth.test_logging"
    assert logger_custom_name.name == "custom_name"
