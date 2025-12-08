import io
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from reviewing_agents.shared.config_mixin import ConfigMixin
from reviewing_agents.shared.llm_interface import call_llm, call_llm_with_image, extract_json
from reviewing_agents.shared.pdf_utils import pdf_first_page_to_image

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


class FormatResponse(BaseModel):
    result: int  # 0 = OK, 1 = format bad
    reasoning: str
    violations: list[str]


class FormatDetectorProtocol(ABC):
    @abstractmethod
    def __init__(self, config: dict):
        pass

    @abstractmethod
    def check_format(self, pdf_file: bytes) -> FormatResponse:
        pass


class LLMFormatDetector(ConfigMixin, FormatDetectorProtocol):
    def check_format(self, pdf_file: bytes) -> FormatResponse:
        response = call_llm(
            system_prompt=self.prompts["system"],
            user_instruction=self.prompts["user"],
            file_data=pdf_file,
            **self.llm_config,
        )

        return extract_json(response, FormatResponse)


class ImageFormatDetector(ConfigMixin, FormatDetectorProtocol):
    def __init__(self, config_override=None, good_example_pdfs: list[bytes] | None = None):
        super().__init__(config_override)
        self._good_example_images: list[bytes] = []

        if good_example_pdfs:
            for pdf_bytes in good_example_pdfs:
                self._good_example_images.append(self._pdf_to_png_bytes(pdf_bytes))
        else:
            default_examples = [
                DATA_DIR / "wrong_proof.pdf",
                DATA_DIR / "hallucinated_ref.pdf",
            ]
            for pdf_path in default_examples:
                image = pdf_first_page_to_image(str(pdf_path))
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                self._good_example_images.append(buf.getvalue())

    def _pdf_to_png_bytes(self, pdf_bytes: bytes) -> bytes:
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_file.flush()
            image = pdf_first_page_to_image(tmp_file.name)
            os.unlink(tmp_file.name)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def check_format(self, pdf_file: bytes) -> FormatResponse:
        submitted_image_bytes = self._pdf_to_png_bytes(pdf_file)

        response = call_llm_with_image(
            system_prompt=self.prompts["system"],
            user_instruction=self.prompts["user"],
            good_images=self._good_example_images,
            test_image=submitted_image_bytes,
            **self.llm_config,
        )

        return extract_json(response, FormatResponse)


if __name__ == "__main__":
    from pathlib import Path

    detector = LLMFormatDetector()

    pdf_path = Path(__file__).parent.parent.parent.parent / "data" / "wrong_proof.pdf"
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()

    result = detector.check_format(pdf_data)
    print(f"Testing LLMFormatDetector with: {pdf_path.name}")
    print(f"Result: {result.result}")
    print(f"Reasoning: {result.reasoning}")
    if result.violations:
        print(f"Violations: {result.violations}")
