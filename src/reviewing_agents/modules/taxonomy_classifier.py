from abc import ABC, abstractmethod

from pydantic import BaseModel

from reviewing_agents.shared.config_mixin import ConfigMixin
from reviewing_agents.shared.llm_interface import call_llm, extract_json


class TaxonomyResponse(BaseModel):
    categories: list[str]


class TaxonomyClassifierProtocol(ABC):
    @abstractmethod
    def classify_paper(self, pdf_file: bytes) -> TaxonomyResponse:
        pass


class ArxivTaxonomyClassifier(ConfigMixin, TaxonomyClassifierProtocol):
    def classify_paper(self, pdf_file: bytes) -> TaxonomyResponse:
        response = call_llm(
            system_prompt=self.prompts["system"],
            user_instruction=self.prompts["user"],
            file_data=pdf_file,
            **self.llm_config,
        )

        return extract_json(response, TaxonomyResponse)


if __name__ == "__main__":
    from pathlib import Path

    classifier = ArxivTaxonomyClassifier()

    pdf_path = Path(__file__).parent.parent.parent.parent / "data" / "wrong_proof.pdf"
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()

    result = classifier.classify_paper(pdf_data)
    print(f"Categories: {result.categories}")
