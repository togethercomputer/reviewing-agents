from abc import ABC, abstractmethod

from pydantic import BaseModel

from reviewing_agents.shared.config_mixin import ConfigMixin
from reviewing_agents.shared.llm_interface import call_llm, extract_json


class CorrectnessResponse(BaseModel):
    score: int
    reasoning: str
    key_issues: list[str]


class CorrectnessDetectorProtocol(ABC):
    @abstractmethod
    def __init__(self, config: dict):
        pass

    @abstractmethod
    def check_correctness(self, pdf_file: bytes) -> CorrectnessResponse:
        pass


class LLMCorrectnessDetector(ConfigMixin, CorrectnessDetectorProtocol):
    def check_correctness(self, pdf_file: bytes) -> CorrectnessResponse:
        response = call_llm(
            system_prompt=self.prompts["system"],
            user_instruction=self.prompts["user"],
            file_data=pdf_file,
            **self.llm_config,
        )

        return extract_json(response, CorrectnessResponse)


class LLMCorrectnessAggregator(ConfigMixin):
    def aggregate(self, responses: list[CorrectnessResponse]) -> CorrectnessResponse:
        responses_text = "\n\n".join(
            [
                f"Response {i + 1}:\nScore: {r.score}\nReasoning: {r.reasoning}\nKey Issues: {r.key_issues}"
                for i, r in enumerate(responses)
            ]
        )

        response = call_llm(
            system_prompt=self.prompts["system"],
            user_instruction=self.prompts["user"].format(responses=responses_text),
            **self.llm_config,
        )

        return extract_json(response, CorrectnessResponse)


if __name__ == "__main__":
    from pathlib import Path

    detector = LLMCorrectnessDetector()

    pdf_path = Path(__file__).parent.parent.parent.parent / "data" / "wrong_proof.pdf"

    with open(pdf_path, "rb") as f:
        pdf_data = f.read()

    result = detector.check_correctness(pdf_data)

    print(f"Score: {result.score}")
    print(f"Reasoning: {result.reasoning}")
    if result.key_issues:
        print(f"Key Issues: {result.key_issues}")
