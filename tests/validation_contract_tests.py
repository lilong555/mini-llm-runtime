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

def precision_experiment_boundaries():
    p = read("benchmarks/runtime-inputs/qwen3-precision-v1.json")
    assert p["protocol_id"] == "precision-experiment-v1" and p["spec_id"] == "CUDA-PREC-001"
    assert p["model_sha256"] == protocol["model_sha256"]
    paths = [p["validation_contract"], *p["serving"]["traces"]]
    for item in paths:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert p["model"]["token_ids"] == protocol["token_ids"]
    assert p["default_mode"] == p["baseline_mode"] == "f32-pedantic"
    assert p["candidate_mode"] == "f16-matrix-f32acc"
    assert p["measurement"]["independent_trials"] == 3
    assert p["measurement"]["warmup"] == 2 and p["measurement"]["measured_repeats"] == 3
    assert p["measurement"]["same_binary_per_layer"] and p["measurement"]["one_precision_per_process"]
    assert p["gpu"] == dict(device=0, max_sequences=4, max_model_len=2048, batch_tokens=128,
                            streams=1, kv_layout="contiguous", kv_dtype="F16")
    micro = p["micro"]
    assert micro["rows"] == [1, 4, 32, 128] and micro["inner_iterations"] == 20
    assert micro["shape_count"] == len(micro["matrices"]) * len(micro["rows"]) == 16
    assert [(x["role"], x["N"], x["K"]) for x in micro["matrices"]] == [
        ("Q", 2048, 1024), ("gate", 3072, 1024), ("down", 1024, 3072), ("LM_head", 151936, 1024)]
    old_micro = read("benchmarks/runtime-inputs/qwen3-cuda-micro-v0.json")["matrix"]
    for matrix in micro["matrices"]:
        assert matrix["seeds"] == [old_micro["roles"].index(matrix["role"]) * len(old_micro["rows"])
                                   + old_micro["rows"].index(m) + 1 for m in micro["rows"]]
    assert micro["accumulation_bound_multiplier"] == 4
    assert micro["fp32_unit_roundoff"] == 2 ** -24
    numerical = p["numerical"]
    assert numerical["configuration_count"] == (len(numerical["corpus_ids"]) * len(numerical["lengths"])
                                                * len(numerical["chunk_tokens"]) * len(numerical["sequence_counts"])) == 48
    assert numerical["lengths"] == [16, 128, 1536]
    assert numerical["chunk_tokens"] == [16, 128] and numerical["sequence_counts"] == [1, 4]
    for key in ("rmse_exclusive", "max_absolute_exclusive", "cosine_min_inclusive", "all_finite"):
        assert numerical[key] == contract["thresholds"][key]
    assert numerical["argmax_require_equal_if"] == contract["argmax"]["require_equal_if"]
    assert len(p["model"]["workloads"]) == 6 and p["model"]["workloads"][0]["name"] == "prefill-128"
    for work in p["model"]["workloads"]:
        if work["mode"] == "fixed_context_decode":
            assert work["effective_length"] == work["prefix_tokens"] + 1
    assert p["gates"]["primary_metric"] == "host_forward_to_token_ns"
    assert p["gates"]["median_gain_min"] == 0.10 and p["gates"]["every_trial_gain_min"] == 0.05
    assert p["gates"]["owned_device_bytes_reduction_min"] == 0.30
    assert p["serving"]["policy"] == "mixed" and p["serving"]["telemetry"] == "off"
    assert p["budget"]["total_performance_processes"] == sum(
        p["budget"][key] for key in ("micro_processes", "model_processes", "serving_processes")) == 24
    assert p["budget"]["additional_trials_to_find_positive_result"] == 0
    assert p["budget"]["new_nsys_captures"] == p["budget"]["ncu_sessions_if_needed"] == 1
    assert not p["promotion"]["default_changes"] and not p["promotion"]["backup_on_success"]


if __name__ == "__main__":
    tests = [pinned_weights, golden_provenance, corpus_boundaries, measurement_boundaries,
             precision_experiment_boundaries]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"{len(tests)}/{len(tests)} tests passed")
