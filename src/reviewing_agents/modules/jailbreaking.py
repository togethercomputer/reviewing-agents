from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from reviewing_agents.shared.config_mixin import ConfigMixin
from reviewing_agents.shared.llm_interface import call_llm
from reviewing_agents.shared.string_utils import extract_answer, extract_reasoning


class AbuseResult(Enum):
    ABUSE = 1
    OK = 2


@dataclass
class AbuseResponse:
    result: AbuseResult
    reasoning: str


class JailbreakingCheckerProtocol(ABC):
    @abstractmethod
    def check_jailbreaking(self, pdf_file: bytes) -> AbuseResponse:
        pass


class JailbreakingChecker(ConfigMixin, JailbreakingCheckerProtocol):
    def check_jailbreaking(self, pdf_file: bytes) -> AbuseResponse:
        response = call_llm(
            system_prompt=self.prompts["system"],
            user_instruction=self.prompts["user"],
            file_data=pdf_file,
            **self.llm_config,
        )
        reasoning = extract_reasoning(response)
        answer = extract_answer(response)

        if not reasoning or not answer:
            raise ValueError("LLM failed to return reasoning or answer")

        if answer and answer.lower() == "abuse":
            return AbuseResponse(AbuseResult.ABUSE, reasoning or "LLM detected potential abuse")
        return AbuseResponse(AbuseResult.OK, reasoning or "LLM found no issues")


def check_if_in_text(check: str, document: str):
    if check.lower() in document.lower():
        return AbuseResponse(AbuseResult.ABUSE, f"Found '{check}' in document")
    return AbuseResponse(AbuseResult.OK, "No concerning text found")


def llm_jailbreaking_check(pdf_file: bytes, config: dict):
    response = call_llm(
        system_prompt=config["prompts"]["system"],
        user_instruction=config["prompts"]["user"],
        file_data=pdf_file,
        **config["llm"],
    )
    reasoning = extract_reasoning(response)
    answer = extract_answer(response)

    if not reasoning or not answer:
        raise ValueError("LLM failed to return reasoning or answer")

    if answer and answer.lower() == "abuse":
        return AbuseResponse(AbuseResult.ABUSE, reasoning or "LLM detected potential abuse")
    return AbuseResponse(AbuseResult.OK, reasoning or "LLM found no issues")


if __name__ == "__main__":
    from pathlib import Path

    checker = JailbreakingChecker()

    pdf_path = Path(__file__).parent.parent.parent.parent / "data" / "jailbreaking_white_text.pdf"
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()

    result = checker.check_jailbreaking(pdf_data)
    print(f"Result: {result.result}")
    print(f"Reasoning: {result.reasoning}")
