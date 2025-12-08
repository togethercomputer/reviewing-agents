from typing import Type, TypeVar

from pydantic import BaseModel

from reviewing_agents.shared.config_mixin import ConfigMixin
from reviewing_agents.shared.llm_interface import call_llm

T = TypeVar("T", bound=BaseModel)


class JsonExtractor(ConfigMixin):
    def extract(self, text: str, response_model: Type[T]) -> T:
        user_prompt = self.prompts["user"].format(text=text)

        response = call_llm(
            system_prompt=self.prompts["system"],
            user_instruction=user_prompt,
            response_model=response_model,
            **self.llm_config,
        )

        return response
