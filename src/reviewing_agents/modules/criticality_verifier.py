from abc import ABC, abstractmethod

from pydantic import BaseModel

from reviewing_agents.shared.config_mixin import ConfigMixin
from reviewing_agents.shared.llm_interface import call_llm, extract_json


class CriticalityIssues(BaseModel):
    major: list[str]
    minor: list[str]
    false_positives: list[str]


class CriticalityResponse(BaseModel):
    score: int
    reasoning: str
    issues: CriticalityIssues


class CriticalityVerifierProtocol(ABC):
    @abstractmethod
    def __init__(self, config: dict):
        pass

    @abstractmethod
    def verify_criticality(self, pdf_file: bytes, correctness_findings: dict) -> CriticalityResponse:
        pass


class LLMCriticalityVerifier(ConfigMixin, CriticalityVerifierProtocol):
    def verify_criticality(self, pdf_file: bytes, correctness_findings: dict) -> CriticalityResponse:
        findings_text = f"""
Original Correctness Score: {correctness_findings["score"]}
Reasoning: {correctness_findings["reasoning"]}
Key Issues Identified:
{chr(10).join(f"- {issue}" for issue in correctness_findings["key_issues"])}
"""

        response = call_llm(
            system_prompt=self.prompts["system"],
            user_instruction=self.prompts["user"].format(findings=findings_text),
            file_data=pdf_file,
            **self.llm_config,
        )

        return extract_json(response, CriticalityResponse)


if __name__ == "__main__":
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from dataclasses import asdict, dataclass
    from datetime import datetime
    from pathlib import Path

    from tqdm import tqdm

    from reviewing_agents.modules.correctness import LLMCorrectnessDetector

    @dataclass
    class CombinedResult:
        filename: str
        correctness_score: int
        correctness_reasoning: str
        correctness_key_issues: list[str]
        criticality_score: int
        criticality_reasoning: str
        major_issues: list[str]
        minor_issues: list[str]
        false_positives: list[str]

    def evaluate_pdf(
        pdf_path: Path,
        correctness_detector: LLMCorrectnessDetector,
        criticality_verifier: LLMCriticalityVerifier,
    ) -> CombinedResult:
        filename = pdf_path.name

        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        correctness_result = correctness_detector.check_correctness(pdf_data)

        correctness_findings = {
            "score": correctness_result.score,
            "reasoning": correctness_result.reasoning,
            "key_issues": correctness_result.key_issues,
        }

        criticality_result = criticality_verifier.verify_criticality(pdf_data, correctness_findings)

        return CombinedResult(
            filename=filename,
            correctness_score=correctness_result.score,
            correctness_reasoning=correctness_result.reasoning,
            correctness_key_issues=correctness_result.key_issues,
            criticality_score=criticality_result.score,
            criticality_reasoning=criticality_result.reasoning,
            major_issues=criticality_result.issues.major,
            minor_issues=criticality_result.issues.minor,
            false_positives=criticality_result.issues.false_positives,
        )

    correctness_detector = LLMCorrectnessDetector()
    criticality_verifier = LLMCriticalityVerifier()

    data_dir = Path(__file__).parent.parent.parent / "data"
    pdf_files = [data_dir / "wrong_proof.pdf"]

    results = []

    with ThreadPoolExecutor(max_workers=15) as executor:
        future_to_pdf = {
            executor.submit(evaluate_pdf, pdf_file, correctness_detector, criticality_verifier): pdf_file
            for pdf_file in pdf_files
        }

        for future in tqdm(as_completed(future_to_pdf), total=len(pdf_files), desc="Processing papers"):
            pdf_file = future_to_pdf[future]
            result = future.result()
            results.append(result)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = Path(f"correctness_criticality_results_{timestamp}.json")

    output_data = {"timestamp": timestamp, "results": [asdict(r) for r in results]}

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
