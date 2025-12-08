from abc import ABC, abstractmethod

from pydantic import BaseModel

from reviewing_agents.shared.config_mixin import ConfigMixin
from reviewing_agents.shared.llm_interface import call_llm


class ExpertiseResponse(BaseModel):
    profile: str


class ExpertiseDetectorProtocol(ABC):
    @abstractmethod
    def detect_expertise(self, pdf_file: bytes) -> str:
        pass


class ExpertiseDetector(ConfigMixin, ExpertiseDetectorProtocol):
    def detect_expertise(self, pdf_file: bytes) -> str:
        response = call_llm(
            system_prompt=self.prompts["system"],
            user_instruction=self.prompts["user"],
            file_data=pdf_file,
            **self.llm_config,
        )

        json_response = call_llm(
            system_prompt="You are a JSON parser.",
            user_instruction="Please parse the following text into a JSON object: " + response,
            response_model=ExpertiseResponse,
            model="gpt-4o-mini",
            temperature=0.0,
        )

        return json_response.profile


def detect_expertise_needed(pdf_file: bytes, config: dict) -> str:
    response = call_llm(
        system_prompt=config["prompts"]["system"],
        user_instruction=config["prompts"]["user"],
        file_data=pdf_file,
        **config["llm"],
    )

    json_response = call_llm(
        system_prompt="You are a JSON parser.",
        user_instruction="Please parse the following text into a JSON object: " + response,
        response_model=ExpertiseResponse,
        model="gpt-4o-mini",
        temperature=0.0,
    )

    return json_response.profile
