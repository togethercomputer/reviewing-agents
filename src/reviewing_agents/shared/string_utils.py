import re


def extract_tag_content(text: str, tag: str) -> str | None:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_reasoning(text: str) -> str | None:
    return extract_tag_content(text, "reasoning")


def extract_answer(text: str) -> str | None:
    return extract_tag_content(text, "answer")
