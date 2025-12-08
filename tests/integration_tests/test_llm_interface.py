import pytest

from reviewing_agents.shared.llm_interface import encode_pdf_data


@pytest.mark.integration
def test_encode_pdf_data():
    pdf_data = b"fake pdf content"
    result = encode_pdf_data(pdf_data)

    assert result.startswith("data:application/pdf;base64,")
    assert len(result) > len("data:application/pdf;base64,")


@pytest.mark.integration
def test_encode_empty_pdf():
    pdf_data = b""
    result = encode_pdf_data(pdf_data)

    assert result == "data:application/pdf;base64,"
