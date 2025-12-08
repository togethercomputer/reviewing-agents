import base64
from typing import Optional, Type, TypeVar, Union, overload

import litellm
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

litellm.drop_params = True


T = TypeVar("T", bound=BaseModel)


def extract_json(text: str, response_model: Type[T]) -> T:
    from reviewing_agents.shared.json_extractor import JsonExtractor

    extractor = JsonExtractor()
    return extractor.extract(text, response_model)


def encode_pdf_data(pdf_data: bytes) -> str:
    encoded_file = base64.b64encode(pdf_data).decode("utf-8")
    return f"data:application/pdf;base64,{encoded_file}"


def encode_image_data(image_data: bytes) -> str:
    encoded_file = base64.b64encode(image_data).decode("utf-8")
    return f"data:image/png;base64,{encoded_file}"


@overload
def call_llm(
    system_prompt: str,
    user_instruction: str,
    model: str = "gpt-5",
    temperature: float = 1,
    max_tokens: Optional[int] = None,
    file_data: Optional[bytes] = None,
    history: Optional[list[dict]] = None,
    response_model: Type[T] = ...,
    **kwargs,
) -> T: ...


@overload
def call_llm(
    system_prompt: str,
    user_instruction: str,
    model: str = "gpt-5",
    temperature: float = 1,
    max_tokens: Optional[int] = None,
    file_data: Optional[bytes] = None,
    history: Optional[list[dict]] = None,
    response_model: None = None,
    **kwargs,
) -> str: ...


def _log_retry(retry_state):
    print(
        f"\n[RETRY] Attempt {retry_state.attempt_number} failed, retrying in {retry_state.next_action.sleep} seconds..."
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=60), before_sleep=_log_retry)
def call_llm(
    system_prompt: str,
    user_instruction: str,
    model: str = "gpt-5",
    temperature: float = 1,
    max_tokens: Optional[int] = None,
    file_data: Optional[bytes] = None,
    history: Optional[list[dict]] = None,
    response_model: Optional[Type[T]] = None,
    **kwargs,
) -> Union[T, str]:
    messages = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history)

    if file_data:
        base64_url = encode_pdf_data(file_data)
        user_content = [
            {"type": "text", "text": user_instruction},
            {
                "type": "file",
                "file": {
                    "file_data": base64_url,
                },
            },
        ]
    else:
        user_content = user_instruction

    messages.append({"role": "user", "content": user_content})

    completion_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "timeout": 1500,
        **kwargs,
    }

    if "gpt" in model.lower():
        completion_kwargs["safety_identifier"] = "reviewer_agent"

    if max_tokens is not None:
        completion_kwargs["max_tokens"] = max_tokens

    if response_model is not None:
        completion_kwargs["response_format"] = response_model

    if response_model is not None:
        response = litellm.completion(**completion_kwargs, drop_params=True)
        content = response.choices[0].message.content
        return response_model.model_validate_json(content)
    else:
        completion_kwargs["stream"] = True
        response_stream = litellm.completion(**completion_kwargs, drop_params=True)
        full_content = ""
        for chunk in response_stream:
            if chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
        return full_content


@overload
def call_llm_with_image(
    system_prompt: str,
    user_instruction: str,
    good_images: list[bytes],
    test_image: bytes,
    model: str = "gpt-5-mini",
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    history: Optional[list[dict]] = None,
    response_model: Type[T] = ...,
    **kwargs,
) -> T: ...


@overload
def call_llm_with_image(
    system_prompt: str,
    user_instruction: str,
    good_images: list[bytes],
    test_image: bytes,
    model: str = "gpt-5-mini",
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    history: Optional[list[dict]] = None,
    response_model: None = None,
    **kwargs,
) -> str: ...


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=60), before_sleep=_log_retry)
def call_llm_with_image(
    system_prompt: str,
    user_instruction: str,
    good_images: list[bytes],
    test_image: bytes,
    model: str = "gpt-5-mini",
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    history: Optional[list[dict]] = None,
    response_model: Optional[Type[T]] = None,
    **kwargs,
) -> Union[T, str]:
    """
    Yes, this is basically the same as call_llm, but with images instead of a file. It is more confusing to
    put things together in one function because we need to check for both image and file data.
    """
    messages = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history)

    user_content = [{"type": "text", "text": user_instruction}]

    # Add good example images with labels
    for good_image in good_images:
        user_content.append({"type": "text", "text": "This is a good paper (reference example):"})
        base64_url = encode_image_data(good_image)
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": base64_url,
                },
            }
        )

    # Add test image with label
    user_content.append({"type": "text", "text": "This is the paper to evaluate:"})
    base64_url = encode_image_data(test_image)
    user_content.append(
        {
            "type": "image_url",
            "image_url": {
                "url": base64_url,
            },
        }
    )

    messages.append({"role": "user", "content": user_content})

    completion_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "timeout": 1500,
        **kwargs,
    }

    if "gpt" in model.lower():
        completion_kwargs["safety_identifier"] = "reviewer_agent"

    if max_tokens is not None:
        completion_kwargs["max_tokens"] = max_tokens

    if response_model is not None:
        completion_kwargs["response_format"] = response_model

    if response_model is not None:
        response = litellm.completion(**completion_kwargs)
        content = response.choices[0].message.content
        return response_model.model_validate_json(content)
    else:
        completion_kwargs["stream"] = True
        response_stream = litellm.completion(**completion_kwargs)
        full_content = ""
        for chunk in response_stream:
            if chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
        return full_content
