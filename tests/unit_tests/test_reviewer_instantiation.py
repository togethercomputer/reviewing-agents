from reviewing_agents.modules.reviewers.simple_reviewer import SimpleReviewer  # This registers the reviewer
from reviewing_agents.shared.config import Config


def test_all_configured_reviewers_instantiation():
    """Test that all configured reviewers can be instantiated without LLM calls"""
    # Import will register the reviewer
    assert SimpleReviewer is not None

    config = Config()
    available_reviewers = config.list_available_reviewers()

    assert len(available_reviewers) > 0, "No reviewers configured"

    for reviewer_name in available_reviewers:
        reviewer = config.create_reviewer(reviewer_name)
        assert reviewer is not None, f"Failed to create reviewer: {reviewer_name}"
        assert hasattr(reviewer, "review_paper"), f"Reviewer {reviewer_name} missing review_paper method"
