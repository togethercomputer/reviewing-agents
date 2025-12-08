from unittest.mock import patch

from reviewing_agents.modules.reference_check.light import ReferenceCheckLight
from reviewing_agents.modules.reference_check.models import Reference, References


def test_reference_model_creation():
    ref = Reference(title="Test Paper", authors="John Doe, Jane Smith", journal="Test Journal")

    assert ref.title == "Test Paper"
    assert ref.authors == "John Doe, Jane Smith"
    assert ref.journal == "Test Journal"


def test_references_model_creation():
    refs = References(
        references=[
            Reference(title="Paper 1", authors="Author 1", journal="Journal 1"),
            Reference(title="Paper 2", authors="Author 2", journal="Journal 2"),
        ]
    )

    assert len(refs.references) == 2
    assert refs.references[0].title == "Paper 1"


def test_reference_check_light_creation():
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.8},
        "prompts": {
            "extraction": {"system": "Test system", "user": "Test user"},
            "hallucination_detection": {"query": "Test query"},
        },
    }

    checker = ReferenceCheckLight(config_override=test_config)
    assert checker.config == test_config
    assert checker.llm_config == test_config["llm"]


def test_reference_check_light_extract_with_mock():
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.8},
        "prompts": {
            "extraction": {"system": "Test system", "user": "Test user"},
            "hallucination_detection": {"query": "Test query"},
        },
    }

    checker = ReferenceCheckLight(config_override=test_config)
    mock_pdf = b"Mock PDF"

    with (
        patch("reviewing_agents.modules.reference_check.light.call_llm") as mock_call_llm,
        patch("reviewing_agents.modules.reference_check.light.extract_json") as mock_extract,
    ):
        mock_extract.return_value = References(references=[])

        result = checker.extract_references(mock_pdf)

        mock_call_llm.assert_called_once()
        assert isinstance(result, References)
