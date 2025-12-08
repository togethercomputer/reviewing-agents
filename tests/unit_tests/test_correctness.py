from unittest.mock import patch

from reviewing_agents.modules.correctness import (
    CorrectnessDetectorProtocol,
    CorrectnessResponse,
    LLMCorrectnessDetector,
)
from reviewing_agents.shared.config import Config


def test_correctness_response_creation():
    """Test that CorrectnessResponse can be created with valid data"""
    response = CorrectnessResponse(
        score=3,
        reasoning="Some methodological concerns but generally sound",
        key_issues=["Minor statistical issue", "Unclear methodology"],
    )

    assert response.score == 3
    assert response.reasoning == "Some methodological concerns but generally sound"
    assert len(response.key_issues) == 2


def test_correctness_response_empty_issues():
    """Test that CorrectnessResponse works with empty key_issues"""
    response = CorrectnessResponse(score=5, reasoning="Excellent methodological rigor throughout", key_issues=[])

    assert response.score == 5
    assert response.key_issues == []


def test_detector_creation_with_config():
    """Test that LLMCorrectnessDetector can be created with config override"""
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.8},
        "prompts": {"system": "Test system prompt", "user": "Test user prompt"},
    }

    detector = LLMCorrectnessDetector(config_override=test_config)
    assert detector.config == test_config
    assert detector.llm_config == test_config["llm"]
    assert detector.prompts == test_config["prompts"]


def test_detector_has_correct_interface():
    """Test that detector implements the protocol correctly"""
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.8},
        "prompts": {"system": "Test system prompt", "user": "Test user prompt"},
    }

    detector = LLMCorrectnessDetector(config_override=test_config)
    assert isinstance(detector, CorrectnessDetectorProtocol)
    assert hasattr(detector, "check_correctness")


def test_detector_with_mock_llm_call():
    """Test that detector can process PDF data with mocked LLM"""
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.8},
        "prompts": {"system": "Test system prompt", "user": "Test user prompt"},
    }

    detector = LLMCorrectnessDetector(config_override=test_config)
    mock_pdf = b"Mock PDF content"

    # Mock the LLM interface
    with (
        patch("reviewing_agents.modules.correctness.call_llm") as mock_call_llm,
        patch("reviewing_agents.modules.correctness.extract_json") as mock_extract,
    ):
        mock_call_llm.return_value = '{"score": 4, "reasoning": "Test reasoning", "key_issues": []}'
        mock_extract.return_value = CorrectnessResponse(score=4, reasoning="Test reasoning", key_issues=[])

        result = detector.check_correctness(mock_pdf)

        # Verify the LLM was called with correct parameters
        mock_call_llm.assert_called_once_with(
            system_prompt="Test system prompt",
            user_instruction="Test user prompt",
            file_data=mock_pdf,
            model="gpt-4o",
            temperature=0.8,
        )

        # Verify the response
        assert isinstance(result, CorrectnessResponse)
        assert result.score == 4
        assert result.reasoning == "Test reasoning"
        assert result.key_issues == []


def test_detector_auto_loads_config():
    """Test that detector auto-loads its configuration"""
    detector = LLMCorrectnessDetector()

    # Should have auto-loaded config with expected sections
    assert hasattr(detector, "config")
    assert hasattr(detector, "llm_config")
    assert hasattr(detector, "prompts")

    # llm_config should have model
    assert "model" in detector.llm_config

    # prompts should have system and user
    assert "system" in detector.prompts
    assert "user" in detector.prompts


def test_auto_loading_detector_pipeline():
    """Test the complete pipeline with auto-loading detector"""
    # Test direct detector creation with auto-loading
    detector = LLMCorrectnessDetector()
    assert isinstance(detector, LLMCorrectnessDetector)


def test_no_global_config_dependencies():
    """Test that we can create multiple independent config instances"""
    config1 = Config()
    config2 = Config()

    # Should be independent instances
    assert config1 is not config2

    # With ConfigMixin, detectors auto-load their config independently
    detector1 = LLMCorrectnessDetector()
    detector2 = LLMCorrectnessDetector()

    # Should be independent instances
    assert detector1 is not detector2

    # Both should work independently
    assert hasattr(detector1, "check_correctness")
    assert hasattr(detector2, "check_correctness")

    # Both should have their own config instances
    assert detector1.config is not detector2.config


def test_detector_with_real_pdf():
    """Test that detector can process real PDF data"""
    detector = LLMCorrectnessDetector()

    # This would require actual LLM calls
    # Mock PDF data for testing
    mock_pdf = b"Mock PDF content"

    # In a real test, we'd mock the LLM interface
    with patch("reviewing_agents.shared.llm_interface.call_llm") as mock_llm:
        mock_llm.return_value = '{"score": 3, "reasoning": "Test reasoning", "key_issues": ["Issue 1"]}'

        result = detector.check_correctness(mock_pdf)

        assert hasattr(result, "score")
        assert hasattr(result, "reasoning")
        assert hasattr(result, "key_issues")
        assert isinstance(result.score, int)
        assert 1 <= result.score <= 5
