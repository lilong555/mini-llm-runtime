"""检查预注册数值语料、权重身份和性能协议的内部一致性，不执行 CUDA。"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


contract = read("tests/data/qwen3_validation_cases.json")
protocol = read("benchmarks/runtime-inputs/qwen3-cuda-v0.json")


def pinned_weights():
    model = read(contract["model"]["manifest"])
    reference = read(contract["reference"]["manifest"])
    assert model["sha256"] == contract["model"]["sha256"] == reference["source_sha256"]
    assert reference["sha256"] == contract["reference"]["sha256"]
    assert reference["source_sha256"] == contract["reference"]["source_sha256"]
    assert reference["converter_commit"] == contract["tokenizer"]["commit"]
    assert protocol["model_sha256"] == model["sha256"]


def golden_provenance():
    source = contract["golden_source"]
    raw = (ROOT / source["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == source["sha256"]
    history = {row["prompt"]: row for row in json.loads(raw)["generations"]}
    assert len(contract["stable_greedy"]) == 3
    for case in contract["stable_greedy"]:
        assert case["input_token_ids"]
        assert len(case["expected_token_ids"]) == source["generation_tokens"] == 8
        previous = history[case["text"]]
        assert case["expected_token_ids"] == previous["reference"] == previous["actual"]


def corpus_boundaries():
    recipe = contract["teacher_forcing"]
    assert recipe["mode"] == "fixed_input_tokens"
    assert recipe["expansion"] == "repeat_seed_then_truncate"
    assert recipe["lengths"] == [16, 33, 128, 256, 1536]
    assert {case["id"] for case in contract["corpus"]} == {"zh", "en", "repeated", "special"}
    vocabulary = contract["model"]["vocabulary"]
    for case in contract["corpus"]:
        seed = case["seed_token_ids"]
        assert seed and all(type(token) is int and 0 <= token < vocabulary for token in seed)
        for length in recipe["lengths"]:
            expanded = (seed * ((length + len(seed) - 1) // len(seed)))[:length]
            positions = [p for p in recipe["positions"] if p < length]
            assert len(expanded) == length and positions[-1] == length - 1
            assert len(set(positions)) == len(positions) and positions[0] == 0
    special = next(case for case in contract["corpus"] if case["id"] == "special")
    assert all(token["type"] == "CONTROL" and token["id"] in special["seed_token_ids"]
               for token in contract["special_tokens"])


def measurement_boundaries():
    measurement = protocol["measurement"]
    assert measurement["independent_trials"] == 5
    assert measurement["warmup"] == 2 and measurement["measured_repeats"] == 3
    assert measurement["profiler"] == "none" and measurement["retain_all_samples"]
    assert measurement["statistics_unit"] == "independent_trial_median"
    gpu = protocol["gpu"]
    assert (gpu["max_sequences"], gpu["max_model_len"], gpu["batch_tokens"]) == (4, 2048, 128)
    assert gpu["page_tokens"] is None and protocol["cpu"]["threads"] == [8, 16]
    names = [case["name"] for case in protocol["workloads"]]
    assert len(set(names)) == len(names) == 12
    for case in protocol["workloads"]:
        if case["mode"] == "prefill" and case["tokens"] > gpu["batch_tokens"]:
            assert case["chunk"] <= gpu["batch_tokens"]
        if case["mode"] == "fixed_context_decode":
            assert case["effective_length"] == case["prefix_tokens"] + 1
        if case["mode"] == "natural_generation":
            assert case["prefill_calls"] + case["decode_calls"] == case["output_tokens"]


if __name__ == "__main__":
    tests = [pinned_weights, golden_provenance, corpus_boundaries, measurement_boundaries]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"{len(tests)}/{len(tests)} tests passed")
