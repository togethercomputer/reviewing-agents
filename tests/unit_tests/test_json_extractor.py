from unittest.mock import patch

from pydantic import BaseModel

from reviewing_agents.shared.json_extractor import JsonExtractor


class MockResponse(BaseModel):
    name: str
    age: int


def test_json_extractor_creation():
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.0},
        "prompts": {"system": "Test system", "user": "Test user: {text}"},
    }

    extractor = JsonExtractor(config_override=test_config)
    assert extractor.config == test_config
    assert extractor.llm_config == test_config["llm"]
    assert extractor.prompts == test_config["prompts"]


def test_json_extractor_auto_loads():
    extractor = JsonExtractor()
    assert isinstance(extractor, JsonExtractor)
    assert hasattr(extractor, "extract")


def test_json_extractor_with_mock():
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.0},
        "prompts": {"system": "Test system", "user": "Extract: {text}"},
    }

    extractor = JsonExtractor(config_override=test_config)
    test_text = "John Doe is 25 years old"

    with patch("reviewing_agents.shared.json_extractor.call_llm") as mock_call_llm:
        mock_response = MockResponse(name="John Doe", age=25)
        mock_call_llm.return_value = mock_response

        result = extractor.extract(test_text, MockResponse)

        mock_call_llm.assert_called_once_with(
            system_prompt="Test system",
            user_instruction="Extract: John Doe is 25 years old",
            response_model=MockResponse,
            model="gpt-4o",
            temperature=0.0,
        )

        assert isinstance(result, MockResponse)
        assert result.name == "John Doe"
        assert result.age == 25


def test_json_extractor_formats_prompt():
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.0},
        "prompts": {"system": "System", "user": "Process this: {text}"},
    }

    extractor = JsonExtractor(config_override=test_config)

    with patch("reviewing_agents.shared.json_extractor.call_llm") as mock_call_llm:
        mock_call_llm.return_value = MockResponse(name="Test", age=30)

        extractor.extract("sample text", MockResponse)

        call_args = mock_call_llm.call_args
        assert call_args[1]["user_instruction"] == "Process this: sample text"
