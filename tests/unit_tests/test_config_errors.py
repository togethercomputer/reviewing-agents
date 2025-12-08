import pytest

from reviewing_agents.shared.config import Config


def test_missing_reviewer_config():
    config = Config()

    with pytest.raises(ValueError, match="Unknown reviewer 'nonexistent'"):
        config.create_reviewer("nonexistent")


def test_get_reviewer_config_missing():
    config = Config()

    # The old get_reviewer_config method has been removed
    # Test that create_reviewer fails for missing reviewer instead
    with pytest.raises(ValueError, match="Unknown reviewer 'missing'"):
        config.create_reviewer("missing")


def test_config_loads_without_error():
    config = Config()

    assert hasattr(config, "_data")
    assert isinstance(config._data, dict)


def test_list_available_reviewers_returns_list():
    config = Config()
    reviewers = config.list_available_reviewers()

    assert isinstance(reviewers, list)
