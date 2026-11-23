"""Tests for the AI/ML API provider of ``soup data generate``.

Covers: CLI registration, the fixed https://api.aimlapi.com endpoint, the
AIMLAPI_API_KEY env var, and that the key never reaches another host
(``--api-base`` and HTTP redirects included).
"""

import contextlib
import json
import re

import httpx
import pytest

AIMLAPI_HOST = "api.aimlapi.com"
SYNTHETIC_KEY = "synthetic-aimlapi-key"


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _install_transport(monkeypatch: pytest.MonkeyPatch, handler) -> list[httpx.Request]:
    """Route ``httpx.stream`` through a MockTransport and record every request."""
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    @contextlib.contextmanager
    def fake_stream(method, url, **kwargs):
        follow = kwargs.pop("follow_redirects", False)
        with httpx.Client(
            transport=httpx.MockTransport(recording), follow_redirects=follow
        ) as client:
            with client.stream(method, url, **kwargs) as response:
                yield response

    monkeypatch.setattr(httpx, "stream", fake_stream)
    return seen


def _ok(request: httpx.Request) -> httpx.Response:
    content = json.dumps([{"instruction": "q", "input": "", "output": "a"}])
    return httpx.Response(
        200, json={"choices": [{"message": {"role": "assistant", "content": content}}]}
    )


def _call(**overrides):
    from soup_cli.data.providers.aimlapi import generate_aimlapi

    kwargs = dict(
        prompt="p", count=1, fmt="alpaca", model_name="openai/gpt-4o-mini",
        temperature=0.0, generation_prompt="gen",
    )
    kwargs.update(overrides)
    return generate_aimlapi(**kwargs)


def test_aimlapi_is_a_valid_provider_and_in_help():
    from typer.testing import CliRunner

    from soup_cli.cli import app
    from soup_cli.commands.generate import VALID_PROVIDERS

    assert "aimlapi" in VALID_PROVIDERS
    result = CliRunner().invoke(app, ["data", "generate", "--help"])
    assert "aimlapi" in _strip_ansi(result.output)


def test_sends_bearer_key_to_exact_aimlapi_host(monkeypatch):
    monkeypatch.setenv("AIMLAPI_API_KEY", SYNTHETIC_KEY)
    seen = _install_transport(monkeypatch, _ok)

    result = _call()

    assert result == [{"instruction": "q", "input": "", "output": "a"}]
    assert len(seen) == 1
    request = seen[0]
    assert request.url.scheme == "https"
    assert request.url.host == AIMLAPI_HOST
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {SYNTHETIC_KEY}"
    body = json.loads(request.content)
    assert body["model"] == "openai/gpt-4o-mini"
    assert body["messages"][0] == {"role": "system", "content": "gen"}


def test_missing_key_raises_without_any_request(monkeypatch):
    monkeypatch.delenv("AIMLAPI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-must-not-be-used")
    seen = _install_transport(monkeypatch, _ok)

    with pytest.raises(ValueError, match="AIMLAPI_API_KEY"):
        _call()
    assert seen == []


def test_redirect_to_foreign_host_is_not_followed(monkeypatch):
    monkeypatch.setenv("AIMLAPI_API_KEY", SYNTHETIC_KEY)

    def redirect(request: httpx.Request) -> httpx.Response:
        if request.url.host == AIMLAPI_HOST:
            return httpx.Response(
                307, headers={"Location": "https://evil.example/v1/chat/completions"}
            )
        return _ok(request)

    seen = _install_transport(monkeypatch, redirect)

    with pytest.raises(ValueError, match="307"):
        _call()
    assert [request.url.host for request in seen] == [AIMLAPI_HOST]
    assert all(SYNTHETIC_KEY not in str(request.headers) for request in seen[1:])


def test_generate_batch_routes_aimlapi_and_ignores_api_base(monkeypatch):
    """``--api-base`` must not redirect the AI/ML API key to another host."""
    from soup_cli.commands.generate import _generate_batch

    monkeypatch.setenv("AIMLAPI_API_KEY", SYNTHETIC_KEY)
    seen = _install_transport(monkeypatch, _ok)

    result = _generate_batch(
        prompt="p", count=1, fmt="alpaca", provider="aimlapi",
        model_name="openai/gpt-4o-mini", api_key="openai-key-must-not-be-used",
        api_base="https://evil.example/v1", temperature=0.0, seed_examples=[],
    )

    assert len(result) == 1
    assert [request.url.host for request in seen] == [AIMLAPI_HOST]
    assert seen[0].headers["Authorization"] == f"Bearer {SYNTHETIC_KEY}"


def test_cli_generate_with_aimlapi_writes_output(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIMLAPI_API_KEY", SYNTHETIC_KEY)
    seen = _install_transport(monkeypatch, _ok)

    result = CliRunner().invoke(
        app,
        [
            "data", "generate", "--prompt", "math", "--count", "1",
            "--provider", "aimlapi", "--model", "openai/gpt-4o-mini",
            "--output", "out.jsonl",
        ],
    )

    assert result.exit_code == 0, _strip_ansi(result.output)
    assert seen and all(request.url.host == AIMLAPI_HOST for request in seen)
    rows = (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert SYNTHETIC_KEY not in _strip_ansi(result.output)
