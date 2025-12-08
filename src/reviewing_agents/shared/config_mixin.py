from abc import ABC
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class ConfigMixin(ABC):
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        if config_override:
            self.config = config_override
        else:
            self.config = self._load_config()

        self.llm_config = self.config["llm"]
        self.prompts = self.config["prompts"]

    @classmethod
    def _load_config(cls) -> Dict[str, Any]:
        config_path = PROJECT_ROOT / "config.yaml"
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        class_name = cls.__name__
        if class_name not in data:
            raise ValueError(f"No configuration section found for class '{class_name}'")

        return data[class_name]
