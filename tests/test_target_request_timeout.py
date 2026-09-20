import json

import pytest

from sia.prompts import build_meta_prompt, build_target_client_setup
from sia.providers import load_provider
from sia.run_setup import TaskFiles


def write_provider(tmp_path, timeout):
    path = tmp_path / "local.json"
    path.write_text(json.dumps({
        "provider_id": "local", "name": "Local CPU model",
        "client_kind": "openai", "base_url": "http://127.0.0.1:8001/v1",
        "api_key_env": "LOCAL_QWEN_API_KEY", "request_timeout_seconds": timeout,
    }))
    return str(path)


def test_local_timeout_reaches_both_prompt_instructions(tmp_path):
    provider = load_provider(write_provider(tmp_path, 300))
    task = TaskFiles("sample", "print('reference')", {}, "Answer the question.")
    prompt = build_meta_prompt(task, "local-qwen", "/work", provider=provider)
    assert "timeout=300.0" in prompt
    assert "finite request timeout of 300 seconds or less" in prompt
    assert "finite request timeout of 60 seconds" not in prompt
    assert "timeout=300.0" in build_target_client_setup(provider, "local-qwen")


@pytest.mark.parametrize("timeout", [0, -1, True, "300"])
def test_invalid_provider_timeout_is_rejected(tmp_path, timeout):
    with pytest.raises(SystemExit, match="positive integer"):
        load_provider(write_provider(tmp_path, timeout))
