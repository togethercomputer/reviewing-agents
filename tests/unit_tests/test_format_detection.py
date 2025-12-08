from unittest.mock import patch

from reviewing_agents.modules.format_detection import FormatResponse, LLMFormatDetector


def test_format_response_creation():
    response = FormatResponse(result=0, reasoning="Format looks good", violations=[])

    assert response.result == 0
    assert response.reasoning == "Format looks good"
    assert response.violations == []


def test_format_detector_creation():
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.8},
        "prompts": {"system": "Test system", "user": "Test user"},
    }

    detector = LLMFormatDetector(config_override=test_config)
    assert detector.config == test_config
    assert detector.llm_config == test_config["llm"]
    assert detector.prompts == test_config["prompts"]


def test_format_detector_auto_loads():
    detector = LLMFormatDetector()
    assert isinstance(detector, LLMFormatDetector)
    assert hasattr(detector, "check_format")


def test_format_detector_with_mock():
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.8},
        "prompts": {"system": "Test system", "user": "Test user"},
    }

    detector = LLMFormatDetector(config_override=test_config)
    mock_pdf = b"Mock PDF"

    with (
        patch("reviewing_agents.modules.format_detection.call_llm") as mock_call_llm,
        patch("reviewing_agents.modules.format_detection.extract_json") as mock_extract,
    ):
        mock_extract.return_value = FormatResponse(result=0, reasoning="Good", violations=[])

        result = detector.check_format(mock_pdf)

        mock_call_llm.assert_called_once()
        assert isinstance(result, FormatResponse)
