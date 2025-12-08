from concurrent.futures import ProcessPoolExecutor, as_completed

from reviewing_agents.modules.reference_check.models import Reference, References, run_verification_worker
from reviewing_agents.shared.config_mixin import ConfigMixin
from reviewing_agents.shared.llm_interface import call_llm, extract_json


class ReferenceCheckLight(ConfigMixin):
    def __init__(self, config_override=None):
        super().__init__(config_override)

    def extract_references(self, pdf_file: bytes) -> References:
        response = call_llm(
            system_prompt=self.prompts["extraction"]["system"],
            user_instruction=self.prompts["extraction"]["user"],
            file_data=pdf_file,
            **self.llm_config,
        )
        return extract_json(response, References)

    def _build_query(self, ref: Reference) -> str:
        template = self.prompts["hallucination_detection"]["query"]
        return f"{template}\n\nReference to verify:\n- {ref.title} by {ref.authors} in {ref.journal}"

    def verify_references(self, references: References) -> list[dict]:
        data_list = [{"reference": ref.model_dump(), "query": self._build_query(ref)} for ref in references.references]

        results = []
        with ProcessPoolExecutor(max_workers=min(len(references.references), 5)) as executor:
            futures = {executor.submit(run_verification_worker, d): d for d in data_list}
            for future in as_completed(futures):
                results.append(future.result())

        return results

    def extract_and_verify(self, pdf_file: bytes) -> dict:
        references = self.extract_references(pdf_file)
        verification_results = self.verify_references(references)
        return {"references": references, "verification_results": verification_results}
