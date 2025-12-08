import pytest
import requests

from reviewing_agents.modules.reviewers.simple_reviewer import *  # noqa: F403
from reviewing_agents.shared.config import Config


@pytest.mark.slow
@pytest.mark.integration
def test_all_reviewers_return_valid_scores():
    """Integration test that actually calls LLM APIs - marked as slow"""
    url = "https://arxiv.org/pdf/2503.09516"
    response = requests.get(url)
    pdf_data = response.content

    config = Config()
    available_reviewers = config.list_available_reviewers()

    test_reviewers = ["simple"]

    for reviewer_name in test_reviewers:
        if reviewer_name in available_reviewers:
            reviewer = config.create_reviewer(reviewer_name)
            result = reviewer.review_paper(pdf_data)

            assert result.score is not None, f"Reviewer {reviewer_name} returned None score"
            assert isinstance(result.score, (int, float)), (
                f"Reviewer {reviewer_name} returned non-numeric score: {type(result.score)}"
            )
            assert 1 <= result.score <= 6, (
                f"Reviewer {reviewer_name} returned score outside valid range: {result.score}"
            )
            assert result.review, f"Reviewer {reviewer_name} provided no review"
            assert isinstance(result.review, str), f"Reviewer {reviewer_name} review is not a string"
