from pathlib import Path

import pytest

from reviewing_agents.modules.correctness import LLMCorrectnessDetector
from reviewing_agents.modules.jailbreaking import AbuseResult, JailbreakingChecker
from reviewing_agents.modules.reference_check.light import ReferenceCheckLight

DATA_DIR = Path(__file__).parent.parent.parent / "src" / "data"


@pytest.fixture
def wrong_proof_pdf():
    with open(DATA_DIR / "wrong_proof.pdf", "rb") as f:
        return f.read()


@pytest.fixture
def jailbreaking_pdf():
    with open(DATA_DIR / "jailbreaking_white_text.pdf", "rb") as f:
        return f.read()


@pytest.fixture
def hallucinated_ref_pdf():
    with open(DATA_DIR / "hallucinated_ref.pdf", "rb") as f:
        return f.read()


@pytest.mark.slow
@pytest.mark.integration
def test_correctness_detector_finds_errors_in_wrong_proof(wrong_proof_pdf):
    detector = LLMCorrectnessDetector()
    result = detector.check_correctness(wrong_proof_pdf)

    assert result.score in [1, 2, 3]
    assert result.score >= 2, "Detector should find errors in wrong_proof.pdf"
    assert result.reasoning
    assert isinstance(result.key_issues, list)


@pytest.mark.slow
@pytest.mark.integration
def test_jailbreaking_checker_detects_white_text(jailbreaking_pdf):
    checker = JailbreakingChecker()
    result = checker.check_jailbreaking(jailbreaking_pdf)

    assert result.result == AbuseResult.ABUSE, "Checker should detect jailbreaking attempt"
    assert result.reasoning


@pytest.mark.slow
@pytest.mark.integration
def test_reference_checker_extracts_references(hallucinated_ref_pdf):
    checker = ReferenceCheckLight()
    references = checker.extract_references(hallucinated_ref_pdf)

    assert len(references.references) > 0, "Should extract at least one reference"
    for ref in references.references:
        assert ref.title
