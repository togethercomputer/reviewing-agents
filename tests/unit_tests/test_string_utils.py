from reviewing_agents.shared.string_utils import extract_answer, extract_reasoning, extract_tag_content


def test_extract_tag_content_basic():
    text = "<test>content here</test>"
    result = extract_tag_content(text, "test")
    assert result == "content here"


def test_extract_tag_content_multiline():
    text = "<test>\nmultiline\ncontent\n</test>"
    result = extract_tag_content(text, "test")
    assert result == "multiline\ncontent"


def test_extract_tag_content_not_found():
    text = "no tags here"
    result = extract_tag_content(text, "test")
    assert result is None


def test_extract_reasoning():
    text = "<reasoning>This is the reasoning</reasoning>"
    result = extract_reasoning(text)
    assert result == "This is the reasoning"


def test_extract_answer():
    text = "<answer>4</answer>"
    result = extract_answer(text)
    assert result == "4"


def test_extract_with_other_tags():
    text = "<other>stuff</other><reasoning>the reason</reasoning><more>content</more>"
    result = extract_reasoning(text)
    assert result == "the reason"
