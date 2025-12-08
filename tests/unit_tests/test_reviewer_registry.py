from unittest.mock import patch

import pytest

from reviewing_agents.modules.reviewers.simple_reviewer import (
    SimpleReviewer,
)
from reviewing_agents.shared.config import Config
from reviewing_agents.shared.reviewer_registry import ReviewerRegistry


class TestReviewerRegistry:
    """Test the reviewer registry system"""

    def test_registry_starts_empty(self):
        """Test that we can check registered reviewers"""
        # Note: reviewers are registered when modules are imported
        # so this tests that the registry pattern works
        reviewers = ReviewerRegistry.list_reviewers()
        assert isinstance(reviewers, list)

    def test_registry_creates_reviewer_with_config(self):
        """Test that registry can create reviewers with proper config"""
        test_config = {
            "llm": {"model": "gpt-4o", "temperature": 0.8},
            "prompts": {"system": "Test system prompt", "user": "Test user prompt"},
        }

        reviewer = ReviewerRegistry.create_reviewer("simple", test_config)
        assert isinstance(reviewer, SimpleReviewer)
        assert reviewer.config == test_config

    def test_registry_raises_error_for_unknown_reviewer(self):
        """Test that registry raises error for unknown reviewer types"""
        with pytest.raises(ValueError, match="Unknown reviewer 'nonexistent'"):
            ReviewerRegistry.create_reviewer("nonexistent", {})


class TestConfig:
    """Test the configuration system"""

    def test_config_loads_from_file(self):
        """Test that config loads from YAML file"""
        config = Config()

        # Should load without error
        reviewers = config.list_available_reviewers()
        assert isinstance(reviewers, list)
        assert len(reviewers) > 0

    def test_config_has_expected_reviewers(self):
        """Test that config contains our expected reviewers"""
        config = Config()
        reviewers = config.list_available_reviewers()

        expected_reviewers = ["simple"]
        for reviewer in expected_reviewers:
            assert reviewer in reviewers, f"Expected reviewer '{reviewer}' not found in config"

    def test_config_can_create_all_reviewers(self):
        """Test that config can create all configured reviewers"""
        config = Config()

        for reviewer_name in config.list_available_reviewers():
            reviewer = config.create_reviewer(reviewer_name)
            assert reviewer is not None
            assert hasattr(reviewer, "review_paper")

    def test_config_raises_error_for_unknown_reviewer(self):
        """Test that config raises error for non-existent reviewer"""
        config = Config()

        with pytest.raises(ValueError, match="Unknown reviewer 'nonexistent'"):
            config.create_reviewer("nonexistent")


class TestReviewerTypes:
    """Test different reviewer implementations"""

    def test_simple_reviewer_creation(self):
        """Test SimpleReviewer can be created with config"""
        config = {
            "llm": {"model": "gpt-4o", "temperature": 0.8},
            "prompts": {"system": "Test system prompt", "user": "Test user prompt"},
        }

        reviewer = SimpleReviewer(config)
        assert reviewer.config == config


class TestEndToEndIntegration:
    """Test the complete integration of config + registry + reviewers"""

    def test_config_to_reviewer_pipeline(self):
        """Test the complete pipeline from config to working reviewer"""
        config = Config()

        # Test simple reviewer
        simple_reviewer = config.create_reviewer("simple")
        assert isinstance(simple_reviewer, SimpleReviewer)

    def test_no_global_config_dependencies(self):
        """Test that we can create multiple independent config instances"""
        config1 = Config()
        config2 = Config()

        # Should be independent instances
        assert config1 is not config2

        # Both should work independently
        reviewer1 = config1.create_reviewer("simple")
        reviewer2 = config2.create_reviewer("simple")

        assert reviewer1 is not reviewer2
        assert isinstance(reviewer1, SimpleReviewer)
        assert isinstance(reviewer2, SimpleReviewer)


class TestReviewerFunctionality:
    """Test actual reviewer functionality (skipped by default)"""

    def test_simple_reviewer_with_mock_pdf(self):
        """Test that simple reviewer can process PDF data"""
        config = Config()
        reviewer = config.create_reviewer("simple")

        # This would require actual LLM calls
        # Mock PDF data for testing
        mock_pdf = b"Mock PDF content"

        # In a real test, we'd mock the LLM interface
        with patch("reviewing_agents.shared.llm_interface.call_llm") as mock_llm:
            mock_llm.return_value = "<review>Test review</review><answer>4</answer>"

            result = reviewer.review_paper(mock_pdf)

            assert hasattr(result, "score")
            assert hasattr(result, "review")
