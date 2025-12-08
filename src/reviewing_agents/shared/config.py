from pathlib import Path

import yaml

from reviewing_agents.shared.reviewer_registry import ReviewerRegistry

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class Config:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = PROJECT_ROOT / "config.yaml"

        with open(config_path, "r") as f:
            self._data = yaml.safe_load(f)

    def list_available_reviewers(self) -> list[str]:
        return ReviewerRegistry.list_reviewers()

    def create_reviewer(self, reviewer_name: str):
        return ReviewerRegistry.create_reviewer(reviewer_name, config=None)
