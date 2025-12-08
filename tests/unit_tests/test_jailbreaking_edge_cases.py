from reviewing_agents.modules.jailbreaking import AbuseResponse, AbuseResult, check_if_in_text


def test_check_if_in_text_found():
    document = "This paper discusses jailbreaking techniques"
    result = check_if_in_text("jailbreaking", document)

    assert result.result == AbuseResult.ABUSE
    assert "jailbreaking" in result.reasoning


def test_check_if_in_text_not_found():
    document = "This is a normal academic paper"
    result = check_if_in_text("malicious", document)

    assert result.result == AbuseResult.OK
    assert "No concerning text found" in result.reasoning


def test_check_if_in_text_case_insensitive():
    document = "This contains JAILBREAKING in caps"
    result = check_if_in_text("jailbreaking", document)

    assert result.result == AbuseResult.ABUSE


def test_abuse_response_creation():
    response = AbuseResponse(AbuseResult.OK, "Test reasoning")

    assert response.result == AbuseResult.OK
    assert response.reasoning == "Test reasoning"
