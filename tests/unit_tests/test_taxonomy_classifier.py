from unittest.mock import patch

from reviewing_agents.modules.taxonomy_classifier import (
    ArxivTaxonomyClassifier,
    TaxonomyClassifierProtocol,
    TaxonomyResponse,
)


def test_taxonomy_response_creation():
    response = TaxonomyResponse(categories=["cs.LG", "cs.AI", "stat.ML"])

    assert response.categories == ["cs.LG", "cs.AI", "stat.ML"]
    assert len(response.categories) == 3


def test_taxonomy_response_single_category():
    response = TaxonomyResponse(categories=["cs.CV"])

    assert response.categories == ["cs.CV"]


def test_taxonomy_response_empty_categories():
    response = TaxonomyResponse(categories=[])

    assert response.categories == []


def test_classifier_creation_with_config():
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.3},
        "prompts": {"system": "Test system prompt", "user": "Test user prompt"},
    }

    classifier = ArxivTaxonomyClassifier(config_override=test_config)
    assert classifier.config == test_config
    assert classifier.llm_config == test_config["llm"]
    assert classifier.prompts == test_config["prompts"]


def test_classifier_has_correct_interface():
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.3},
        "prompts": {"system": "Test system prompt", "user": "Test user prompt"},
    }

    classifier = ArxivTaxonomyClassifier(config_override=test_config)
    assert isinstance(classifier, TaxonomyClassifierProtocol)
    assert hasattr(classifier, "classify_paper")


def test_classifier_auto_loads_config():
    classifier = ArxivTaxonomyClassifier()

    assert hasattr(classifier, "config")
    assert hasattr(classifier, "llm_config")
    assert hasattr(classifier, "prompts")
    assert "model" in classifier.llm_config
    assert "system" in classifier.prompts
    assert "user" in classifier.prompts


def test_classifier_with_mock_llm_call():
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.3},
        "prompts": {"system": "Classify this paper", "user": "Please classify"},
    }

    classifier = ArxivTaxonomyClassifier(config_override=test_config)
    mock_pdf = b"Mock PDF content"

    with (
        patch("reviewing_agents.modules.taxonomy_classifier.call_llm") as mock_call_llm,
        patch("reviewing_agents.modules.taxonomy_classifier.extract_json") as mock_extract,
    ):
        mock_extract.return_value = TaxonomyResponse(categories=["cs.LG", "cs.AI"])

        result = classifier.classify_paper(mock_pdf)

        mock_call_llm.assert_called_once_with(
            system_prompt="Classify this paper",
            user_instruction="Please classify",
            file_data=mock_pdf,
            model="gpt-4o",
            temperature=0.3,
        )

        assert isinstance(result, TaxonomyResponse)
        assert result.categories == ["cs.LG", "cs.AI"]


def test_classifier_returns_valid_arxiv_categories():
    test_config = {
        "llm": {"model": "gpt-4o", "temperature": 0.3},
        "prompts": {"system": "Test", "user": "Test"},
    }

    classifier = ArxivTaxonomyClassifier(config_override=test_config)
    mock_pdf = b"Mock PDF"

    valid_categories = ["cs.LG", "cs.CV", "stat.ML", "math.OC"]

    with (
        patch("reviewing_agents.modules.taxonomy_classifier.call_llm"),
        patch("reviewing_agents.modules.taxonomy_classifier.extract_json") as mock_extract,
    ):
        mock_extract.return_value = TaxonomyResponse(categories=valid_categories)

        result = classifier.classify_paper(mock_pdf)

        assert all(cat in result.categories for cat in valid_categories)


def test_multiple_classifier_instances_independent():
    classifier1 = ArxivTaxonomyClassifier()
    classifier2 = ArxivTaxonomyClassifier()

    assert classifier1 is not classifier2
    assert classifier1.config is not classifier2.config
    assert hasattr(classifier1, "classify_paper")
    assert hasattr(classifier2, "classify_paper")
