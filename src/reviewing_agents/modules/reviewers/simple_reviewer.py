from pydantic import BaseModel

from reviewing_agents.shared.config_mixin import ConfigMixin
from reviewing_agents.shared.llm_interface import call_llm, extract_json
from reviewing_agents.shared.reviewer_registry import ReviewerProtocol, ReviewerRegistry


class ReviewResponse(BaseModel):
    score: int
    review: str


@ReviewerRegistry.register("simple")
class SimpleReviewer(ConfigMixin, ReviewerProtocol):
    def review_paper(self, pdf_file: bytes) -> ReviewResponse:
        response = call_llm(
            system_prompt=self.prompts["system"],
            user_instruction=self.prompts["user"],
            file_data=pdf_file,
            **self.llm_config,
        )

        return extract_json(response, ReviewResponse)


@ReviewerRegistry.register("simple_gemini")
class SimpleReviewerGemini(ConfigMixin, ReviewerProtocol):
    def review_paper(self, pdf_file: bytes) -> ReviewResponse:
        response = call_llm(
            system_prompt=self.prompts["system"],
            user_instruction=self.prompts["user"],
            file_data=pdf_file,
            **self.llm_config,
        )

        return extract_json(response, ReviewResponse)


@ReviewerRegistry.register("simple_claude")
class SimpleReviewerClaude(ConfigMixin, ReviewerProtocol):
    def review_paper(self, pdf_file: bytes) -> ReviewResponse:
        response = call_llm(
            system_prompt=self.prompts["system"],
            user_instruction=self.prompts["user"],
            file_data=pdf_file,
            **self.llm_config,
        )

        return extract_json(response, ReviewResponse)


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    reviewers = [
        ("simple", SimpleReviewer()),
        ("simple_gemini", SimpleReviewerGemini()),
        ("simple_claude", SimpleReviewerClaude()),
    ]

    test_pdf_path = Path(__file__).parent.parent.parent.parent.parent / "data" / "wrong_proof.pdf"

    with open(test_pdf_path, "rb") as f:
        pdf_data = f.read()

    print(f"Testing all reviewers concurrently with {test_pdf_path.name}\n")

    def run_reviewer(name, reviewer, pdf_data):
        result = reviewer.review_paper(pdf_data)
        return name, result

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_reviewer, name, reviewer, pdf_data): name for name, reviewer in reviewers}

        for future in as_completed(futures):
            name, result = future.result()
            print(f"{'=' * 60}")
            print(f"Reviewer: {name}")
            print(f"{'=' * 60}")
            print(f"Score: {result.score}")
            print(f"Review: {result.review}")
            print()
