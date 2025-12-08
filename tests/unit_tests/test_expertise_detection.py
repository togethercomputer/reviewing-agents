from unittest.mock import patch

from reviewing_agents.modules.reviewers.expertise_detection import ExpertiseDetector, ExpertiseResponse


def test_expertise_response_creation():
    response = ExpertiseResponse(profile="Machine learning expert with focus on neural networks")
    assert response.profile == "Machine learning expert with focus on neural networks"


def test_expertise_detector_creation():
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.5},
        "prompts": {"system": "Test system", "user": "Test user"},
    }

    detector = ExpertiseDetector(config_override=test_config)
    assert detector.config == test_config
    assert detector.llm_config == test_config["llm"]
    assert detector.prompts == test_config["prompts"]


def test_expertise_detector_auto_loads():
    detector = ExpertiseDetector()
    assert isinstance(detector, ExpertiseDetector)
    assert hasattr(detector, "detect_expertise")


def test_expertise_detector_with_mock():
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.5},
        "prompts": {"system": "Analyze expertise", "user": "Find expertise"},
    }

    detector = ExpertiseDetector(config_override=test_config)
    mock_pdf = b"Mock PDF content about neural networks"

    with patch("reviewing_agents.modules.reviewers.expertise_detection.call_llm") as mock_call_llm:
        mock_call_llm.side_effect = [
            "Expert in machine learning with neural network focus",
            ExpertiseResponse(profile="Machine learning expert with neural network specialization"),
        ]

        result = detector.detect_expertise(mock_pdf)

        assert mock_call_llm.call_count == 2

        first_call = mock_call_llm.call_args_list[0]
        assert first_call[1]["system_prompt"] == "Analyze expertise"
        assert first_call[1]["user_instruction"] == "Find expertise"
        assert first_call[1]["file_data"] == mock_pdf
        assert first_call[1]["model"] == "gpt-4o"
        assert first_call[1]["temperature"] == 0.5

        second_call = mock_call_llm.call_args_list[1]
        assert second_call[1]["system_prompt"] == "You are a JSON parser."
        assert "Please parse the following text into a JSON object" in second_call[1]["user_instruction"]
        assert second_call[1]["response_model"] == ExpertiseResponse

        assert result == "Machine learning expert with neural network specialization"


def test_expertise_detector_has_protocol_interface():
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.5},
        "prompts": {"system": "Test", "user": "Test"},
    }

    detector = ExpertiseDetector(config_override=test_config)
    assert hasattr(detector, "detect_expertise")
    assert callable(detector.detect_expertise)
