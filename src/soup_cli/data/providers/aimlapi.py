"""AI/ML API provider for synthetic data generation.

AI/ML API (https://aimlapi.com) serves many hosted chat models behind one
OpenAI-compatible ``/v1/chat/completions`` endpoint.

Uses raw httpx requests to avoid SDK dependency.
API key must be set via AIMLAPI_API_KEY environment variable. The endpoint is
fixed to https://api.aimlapi.com and is not affected by ``--api-base``, so the
key is only ever sent to that host. Redirects are not followed.
"""

import logging
import os

logger = logging.getLogger(__name__)

AIMLAPI_API_URL = "https://api.aimlapi.com/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def generate_aimlapi(
    prompt: str,
    count: int,
    fmt: str,
    model_name: str,
    temperature: float,
    generation_prompt: str,
) -> list[dict]:
    """Generate examples using the AI/ML API chat completions endpoint.

    Args:
        prompt: User-provided topic/instructions.
        count: Number of examples to generate.
        fmt: Output format (alpaca/sharegpt/chatml).
        model_name: AI/ML API model ID (e.g. ``openai/gpt-4o-mini``).
        temperature: Sampling temperature.
        generation_prompt: Pre-built generation prompt.

    Returns:
        List of generated example dicts.

    Raises:
        ValueError: If API key not found or API returns error.
        ImportError: If httpx is not installed.
    """
    try:
        import httpx  # noqa: F401 -- retain the optional-dependency diagnostic
    except ImportError:
        raise ImportError(
            "httpx is required for AI/ML API generation. Install: pip install httpx"
        )

    api_key = os.environ.get("AIMLAPI_API_KEY")
    if not api_key:
        raise ValueError(
            "AI/ML API key not found. Set AIMLAPI_API_KEY environment variable."
        )

    from soup_cli.utils.http_response import GENERATION_RESPONSE_LIMIT_BYTES, post_capped

    response = post_capped(
        AIMLAPI_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": generation_prompt},
                {"role": "user", "content": f"Generate {count} training examples now."},
            ],
            "temperature": temperature,
            "max_tokens": 4096,
        },
        max_bytes=GENERATION_RESPONSE_LIMIT_BYTES,
        endpoint_label="AI/ML API --provider aimlapi",
        timeout=120.0,
        follow_redirects=False,
    )

    if response.status_code != 200:
        raise ValueError(
            f"AI/ML API returned {response.status_code}. "
            "Check your AIMLAPI_API_KEY and model name."
        )

    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected AI/ML API response format: {exc}") from exc

    from soup_cli.data.providers._utils import parse_json_array

    return parse_json_array(content or "")
