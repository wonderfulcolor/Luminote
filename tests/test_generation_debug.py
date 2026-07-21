from __future__ import annotations

import json

from luminote.generation_debug import GenerationIntegrityStore
from luminote.logging import JsonlLogger
from luminote.model import infer_stop_reason


def test_stop_reason_distinguishes_eos_limit_and_other_stop() -> None:
    assert infer_stop_reason([9, 2], {2}, 20) == "eos"
    assert infer_stop_reason(list(range(20)), {99}, 20) == "maximum_token_limit"
    assert infer_stop_reason([9, 8], {2}, 20) == "configured_stop_token"


def test_generation_integrity_store_records_exact_match_and_mismatch(tmp_path) -> None:
    path = tmp_path / "generation.jsonl"
    store = GenerationIntegrityStore(JsonlLogger(path))
    base = {
        "input_messages": [{"role": "user", "content": "hello"}],
        "rendered_chat_template_prompt": "<prompt>",
        "generation_parameters": {"max_new_tokens": 10},
        "generated_token_ids": [1, 2],
        "server_decoded_text": "Exact text",
        "generated_token_count": 2,
        "stop_reason": "eos",
    }

    store.record_generation("match-id", base)
    assert store.record_frontend_text("match-id", "Exact text") is True
    store.record_generation("mismatch-id", base)
    assert store.record_frontend_text("mismatch-id", "Changed text") is False
    assert store.record_frontend_text("missing-id", "Exact text") is None

    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[0]["event"] == "generation"
    assert events[0]["generated_token_ids"] == [1, 2]
    assert events[1]["server_frontend_text_match"] is True
    assert events[3]["server_frontend_text_match"] is False
