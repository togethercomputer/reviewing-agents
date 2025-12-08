from unittest.mock import patch

from reviewing_agents.modules.criticality_verifier import (
    CriticalityIssues,
    CriticalityResponse,
    CriticalityVerifierProtocol,
    LLMCriticalityVerifier,
)


def test_criticality_issues_creation():
    issues = CriticalityIssues(
        major=["Major error in proof"],
        minor=["Minor typo"],
        false_positives=["Not an actual error"],
    )

    assert issues.major == ["Major error in proof"]
    assert issues.minor == ["Minor typo"]
    assert issues.false_positives == ["Not an actual error"]


def test_criticality_response_creation():
    issues = CriticalityIssues(major=[], minor=["Minor issue"], false_positives=[])
    response = CriticalityResponse(
        score=2,
        reasoning="Found minor issues only",
        issues=issues,
    )

    assert response.score == 2
    assert response.reasoning == "Found minor issues only"
    assert response.issues.minor == ["Minor issue"]


def test_criticality_response_empty_issues():
    issues = CriticalityIssues(major=[], minor=[], false_positives=[])
    response = CriticalityResponse(score=1, reasoning="No genuine errors found", issues=issues)

    assert response.score == 1
    assert response.issues.major == []
    assert response.issues.minor == []
    assert response.issues.false_positives == []


def test_verifier_creation_with_config():
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.5},
        "prompts": {"system": "Test system prompt", "user": "Test user prompt {findings}"},
    }

    verifier = LLMCriticalityVerifier(config_override=test_config)
    assert verifier.config == test_config
    assert verifier.llm_config == test_config["llm"]
    assert verifier.prompts == test_config["prompts"]


def test_verifier_has_correct_interface():
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.5},
        "prompts": {"system": "Test system prompt", "user": "Test user prompt {findings}"},
    }

    verifier = LLMCriticalityVerifier(config_override=test_config)
    assert isinstance(verifier, CriticalityVerifierProtocol)
    assert hasattr(verifier, "verify_criticality")


def test_verifier_auto_loads_config():
    verifier = LLMCriticalityVerifier()

    assert hasattr(verifier, "config")
    assert hasattr(verifier, "llm_config")
    assert hasattr(verifier, "prompts")
    assert "model" in verifier.llm_config
    assert "system" in verifier.prompts
    assert "user" in verifier.prompts


def test_verifier_with_mock_llm_call():
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.5},
        "prompts": {"system": "Test system prompt", "user": "Verify these findings: {findings}"},
    }

    verifier = LLMCriticalityVerifier(config_override=test_config)
    mock_pdf = b"Mock PDF content"
    correctness_findings = {
        "score": 3,
        "reasoning": "Found some issues",
        "key_issues": ["Issue 1", "Issue 2"],
    }

    with (
        patch("reviewing_agents.modules.criticality_verifier.call_llm") as mock_call_llm,
        patch("reviewing_agents.modules.criticality_verifier.extract_json") as mock_extract,
    ):
        mock_issues = CriticalityIssues(major=[], minor=["Issue 1"], false_positives=["Issue 2"])
        mock_extract.return_value = CriticalityResponse(
            score=2, reasoning="One minor issue, one false positive", issues=mock_issues
        )

        result = verifier.verify_criticality(mock_pdf, correctness_findings)

        mock_call_llm.assert_called_once()
        call_kwargs = mock_call_llm.call_args
        assert call_kwargs.kwargs["file_data"] == mock_pdf
        assert call_kwargs.kwargs["system_prompt"] == "Test system prompt"

        assert isinstance(result, CriticalityResponse)
        assert result.score == 2
        assert result.issues.minor == ["Issue 1"]
        assert result.issues.false_positives == ["Issue 2"]


def test_verifier_findings_formatting():
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.5},
        "prompts": {"system": "Test", "user": "{findings}"},
    }

    verifier = LLMCriticalityVerifier(config_override=test_config)
    mock_pdf = b"Mock PDF"
    findings = {
        "score": 3,
        "reasoning": "Test reasoning",
        "key_issues": ["Error A", "Error B"],
    }

    with (
        patch("reviewing_agents.modules.criticality_verifier.call_llm") as mock_call_llm,
        patch("reviewing_agents.modules.criticality_verifier.extract_json") as mock_extract,
    ):
        mock_issues = CriticalityIssues(major=[], minor=[], false_positives=[])
        mock_extract.return_value = CriticalityResponse(score=1, reasoning="All ok", issues=mock_issues)

        verifier.verify_criticality(mock_pdf, findings)

        user_instruction = mock_call_llm.call_args.kwargs["user_instruction"]
        assert "3" in user_instruction
        assert "Test reasoning" in user_instruction
        assert "Error A" in user_instruction
        assert "Error B" in user_instruction
