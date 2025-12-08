import pytest

from reviewing_agents.shared.config_mixin import ConfigMixin


class DummyConfigUser(ConfigMixin):
    pass


class ExistingConfigUser(ConfigMixin):
    @classmethod
    def _load_config(cls):
        return {
            "llm": {"model": "default-model", "temperature": 0.5},
            "prompts": {"system": "default system prompt"},
        }


def test_config_override_replaces_loaded_config():
    override = {
        "llm": {"model": "override-model", "temperature": 0.9},
        "prompts": {"system": "override prompt", "user": "custom user"},
    }

    instance = ExistingConfigUser(config_override=override)

    assert instance.config == override
    assert instance.llm_config["model"] == "override-model"
    assert instance.llm_config["temperature"] == 0.9
    assert instance.prompts["system"] == "override prompt"
    assert instance.prompts["user"] == "custom user"


def test_config_without_override_loads_from_yaml():
    instance = ExistingConfigUser()

    assert instance.llm_config["model"] == "default-model"
    assert instance.prompts["system"] == "default system prompt"


def test_missing_class_in_config_raises_error():
    with pytest.raises(ValueError, match="No configuration section found for class 'DummyConfigUser'"):
        DummyConfigUser()


def test_config_override_exposes_llm_and_prompts():
    override = {
        "llm": {"model": "test-model"},
        "prompts": {"extraction": {"system": "extract stuff"}},
    }

    instance = ExistingConfigUser(config_override=override)

    assert instance.llm_config == {"model": "test-model"}
    assert instance.prompts == {"extraction": {"system": "extract stuff"}}
