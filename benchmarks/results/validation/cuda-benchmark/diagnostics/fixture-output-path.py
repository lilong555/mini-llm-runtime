"""模型基准协议、统计单位与非破坏性复核的确定性反例；不需要 CUDA 或模型。"""

from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_cuda_benchmark as audit

SPEC = audit.read(ROOT / "benchmarks/runtime-inputs/qwen3-cuda-v0.json")


def invalid(function):
    try:
        function()
    except audit.ValidationError:
        return
    raise AssertionError("无效报告未被拒绝")


def fixture(backend="cpu8"):
    dims = dict(embedding=1024, layers=28, heads=16, kv_heads=8, head_dim=128, feed_forward=3072,
                vocabulary=151936, trained_context=40960)
    report = dict(schema_version=1, benchmark=audit.BENCHMARK, status="passed", backend=backend,
                  input_sha256=audit.INPUT_SHA256, model_sha256=audit.MODEL_SHA256, protocol=deepcopy(audit.PROTOCOL),
                  runtime=dict(dimensions=dims, configuration=dict(
                      max_sequences=4, max_model_len=2048, batch_tokens=128, context_pool_tokens=8192,
                      threads=int(backend[3:]), kernel="auto", effective_kernel="avx2-fma-f16c",
                      device=None, streams=None, kv_layout="paged", page_tokens=16),
                      device=None, weights=None, arithmetic=dict(
                          source_weight_dtype="Q8_0", effective_weight_dtype="F32", activation_dtype="F32",
                          kv_dtype="F16", softmax_denominator_dtype="F64", kernel="auto"),
                      initialization=dict(runtime_constructor_ns=1000, model_load_ns=None,
                                          storage_initialization_ns=None, weight_decode_upload_ns=None)),
                  workloads=[])
    allocations = dict(allocation_calls=0, allocations=0, release_calls=0, releases=0, allocated_bytes=0)
    report["before_initialization_allocations"] = allocations
    lengths, resident_pages = [0] * 4, 0

    def snapshot():
        return dict(allocation_stats=deepcopy(allocations), cuda=None, cpu=dict(
            used_kv_pages=sum((v + 15) // 16 for v in lengths),
            resident_kv_bytes=resident_pages * 16 * 28 * 2 * 8 * 128 * 2))

    def call(batch, phase):
        nonlocal resident_pages
        before = lengths[:]
        samples = []
        for index, (token, position, sequence, logits) in enumerate(batch):
            assert lengths[sequence] == position
            lengths[sequence] += 1
            if logits:
                samples.append(dict(sequence=sequence, input_index=index, token=(token + 1) % 151936))
        resident_pages = max(resident_pages, sum((v + 15) // 16 for v in lengths))
        return dict(phase=phase, input_sha256=audit.input_digest(batch), input_tokens=len(batch),
                    logits_rows=len(samples), context_before=before, context_after=lengths[:],
                    host_forward_to_token_ns=100, device_elapsed_ms=None, samples=samples)

    report["initial_state"] = snapshot()
    for work, setup, measured in audit.workload_plan(SPEC):
        actual = dict(name=work["name"], mode=work["mode"], status="passed", iterations=[])
        for iteration in range(5):
            lengths[:] = [0] * 4
            row = dict(index=iteration, phase="warmup" if iteration < 2 else "measured",
                       clear_sequences=[0, 1, 2, 3], clear_ns=10, after_clear=snapshot())
            row["setup"] = [call(batch, "setup") for batch in setup]
            row["before_measured"] = snapshot()
            mode = work["mode"]
            phase = "prefill" if mode in ("prefill", "natural_generation") else "mixed" if mode == "mixed" else "decode"
            row["forwards"] = [call(batch, phase) for batch in measured]
            if mode == "natural_generation":
                for step in range(31):
                    token = row["forwards"][-1]["samples"][0]["token"]
                    row["forwards"].append(call([(token, work["prompt_tokens"] + step, 0, 1)], "decode"))
            row["after_measured"] = snapshot()
            row["setup_forward_ns"] = 100 * len(setup)
            row["host_forward_to_token_ns"] = 100 * len(row["forwards"])
            row["prefill_forward_ns"] = 100 * sum(v["phase"] == "prefill" for v in row["forwards"])
            row["decode_forward_ns"] = 100 * sum(v["phase"] == "decode" for v in row["forwards"])
            row["token_ids"] = [s["token"] for v in row["forwards"] for s in v["samples"]]
            actual["iterations"].append(row)
        report["workloads"].append(actual)
    lengths[:] = [0] * 4
    report["final_state"] = snapshot()
    return report


BASE = fixture()


def cuda_fixture():
    report = deepcopy(BASE)
    report["backend"] = "cuda"
    runtime = report["runtime"]
    runtime["configuration"].update(threads=None, kernel=None, effective_kernel=None, device=0,
                                    streams=1, kv_layout="contiguous", page_tokens=None)
    runtime["device"] = dict(name="fixture", uuid="00000000-0000-0000-0000-000000000000",
                             compute_capability=[8, 9], driver_version=12080, runtime_version=12080, cublas_version=120800)
    runtime["initialization"] = dict(runtime_constructor_ns=2000, model_load_ns=100,
                                     storage_initialization_ns=1800, weight_decode_upload_ns=1500)
    runtime["arithmetic"] = dict(source_weight_dtype="Q8_0", source_tensor_counts=dict(Q8_0=197, F32=113),
        device_weight_dtype="F32", activation_dtype="F32", kv_dtype="F16", kv_rounding="nearest_even",
        qk_pv_accumulation_dtype="F32", softmax_exponential_dtype="F32", softmax_denominator_dtype="F64",
        gemm_compute="CUBLAS_COMPUTE_32F_PEDANTIC", fast_math=False)
    weights, end = [], 0

    def weight(name, rows, columns, kind):
        nonlocal end
        offset, size = (end + 255) // 256 * 256, rows * columns * 4
        weights.append(dict(name=name, source_dtype=kind, device_dtype="F32", shape=[rows, columns],
                            offset=offset, bytes=size, alias_of="", effective_sha256=hashlib.sha256(name.encode()).hexdigest()))
        end = offset + size

    weight("token_embd.weight", 151936, 1024, "Q8_0")
    weights.append(dict(weights[0], name="output.weight", alias_of="token_embd.weight"))
    weight("output_norm.weight", 1, 1024, "F32")
    for layer in range(28):
        for name, rows, columns, kind in (
            ("attn_norm", 1, 1024, "F32"), ("attn_q_norm", 1, 128, "F32"), ("attn_k_norm", 1, 128, "F32"),
            ("ffn_norm", 1, 1024, "F32"), ("attn_q", 2048, 1024, "Q8_0"), ("attn_k", 1024, 1024, "Q8_0"),
            ("attn_v", 1024, 1024, "Q8_0"), ("attn_output", 1024, 2048, "Q8_0"),
            ("ffn_gate", 3072, 1024, "Q8_0"), ("ffn_up", 3072, 1024, "Q8_0"), ("ffn_down", 1024, 3072, "Q8_0")):
            weight(f"blk.{layer}.{name}.weight", rows, columns, kind)
    runtime["weights"] = weights
    plan, payload, _ = audit.memory_plan(runtime)
    assert plan["total_owned_bytes"] == 3449229312 and plan["kv_bytes"] == 939524096 and plan["padding_bytes"] == 488
    counts = dict(inputs=0, rows=0, calls=0)

    def snapshot(lengths):
        state = dict(state="ready", sequence_lengths=lengths[:], live_sequences=sum(v > 0 for v in lengths),
            live_kv_tokens=sum(lengths), kv_capacity_tokens=8192, resident=plan, weight_h2d_bytes=payload,
            rope_h2d_bytes=plan["rope_bytes"], metadata_h2d_bytes=counts["inputs"] * 12 + counts["rows"] * 4,
            token_d2h_bytes=counts["rows"] * 4, status_d2h_bytes=counts["calls"] * 8,
            debug_d2h_bytes=0, intermediate_h2d_bytes=0, intermediate_d2h_bytes=0,
            owned_device_allocations=4, owned_device_bytes=plan["total_owned_bytes"],
            completed_forwards=counts["calls"], post_launch_failures=0, model_load_ns=100,
            storage_initialization_ns=1800, weight_decode_upload_ns=1500)
        return dict(cpu=None, cuda=state, allocation_stats=dict(allocation_calls=4, allocations=4,
                    release_calls=0, releases=0, allocated_bytes=plan["total_owned_bytes"]))

    report["initial_state"] = snapshot([0] * 4)
    for work in report["workloads"]:
        for row in work["iterations"]:
            row["after_clear"] = snapshot([0] * 4)
            lengths = [0] * 4
            for phase, field in (("setup", "before_measured"), ("forwards", "after_measured")):
                for call in row[phase]:
                    counts["inputs"] += call["input_tokens"]
                    counts["rows"] += call["logits_rows"]
                    counts["calls"] += 1
                    lengths = call["context_after"]
                row[field] = snapshot(lengths)
    report["final_state"] = snapshot([0] * 4)
    return report


GPU = cuda_fixture()


def frozen_schedule_is_separate_balanced_and_complete():
    schedule = audit.schedule()
    assert len(schedule) == 70 and len({r["file"] for r in schedule}) == 70
    assert [r["order"] for r in schedule] == list(range(70))
    assert all(r["group"].startswith("aa_") for r in schedule[:30])
    for cpu in ("cpu8", "cpu16"):
        group = [r for r in schedule if r["group"] == f"{cpu}_cuda"]
        for trial in range(5):
            subset = [r for r in group if r["trial"] == trial]
            assert [r["arm"] for r in subset] == (["A", "B", "B", "A"] if trial % 2 == 0 else ["B", "A", "A", "B"])
            assert sorted(r["backend"] for r in subset) == sorted([cpu, cpu, "cuda", "cuda"])


def complete_workload_report_has_exact_counts():
    result = audit.validate_report(BASE, SPEC)
    assert len(result["cases"]) == 12 and result["forwards"] == 585
    assert result["input_tokens"] == 30125 and result["logits_rows"] == 400
    for name, data in result["cases"].items():
        assert len(data["samples"]) == 3
        if name.startswith("generate"):
            assert len(data["token_ids"]) == 32 and data["prefill_samples"] == [100] * 3
            assert data["decode_samples"] == [3100] * 3
    assert BASE["final_state"]["cpu"]["used_kv_pages"] == 0
    assert BASE["final_state"]["cpu"]["resident_kv_bytes"] > 0


def missing_warmup_setup_and_cases_are_rejected():
    for change in ("case", "warmup", "setup", "clear"):
        value = deepcopy(BASE)
        if change == "case":
            value["workloads"].pop()
        elif change == "warmup":
            value["workloads"][0]["iterations"].pop(0)
        elif change == "setup":
            value["workloads"][6]["iterations"][2]["setup"].pop()
        else:
            value["workloads"][6]["iterations"][2]["clear_sequences"] = []
        invalid(lambda: audit.validate_report(value, SPEC))


def changed_inputs_and_natural_history_are_rejected():
    for change in ("input", "position", "sequence", "generation"):
        value = deepcopy(BASE)
        call = value["workloads"][10 if change == "generation" else 0]["iterations"][2]["forwards"][1 if change == "generation" else 0]
        if change in ("input", "generation"):
            call["input_sha256"] = "0" * 64
        elif change == "position":
            call["context_before"][0] = 1
        else:
            call["samples"][0]["sequence"] = 1
        invalid(lambda: audit.validate_report(value, SPEC))


def no_boolean_nonfinite_or_excluded_sampling_time():
    for bad in (True, -1, 0, 100.0, float("nan"), float("inf")):
        value = deepcopy(BASE)
        value["workloads"][0]["iterations"][0]["forwards"][0]["host_forward_to_token_ns"] = bad
        invalid(lambda: audit.validate_report(value, SPEC))
    for change in ("events", "total", "sampling"):
        value = deepcopy(BASE)
        if change == "events":
            value["workloads"][0]["iterations"][0]["forwards"][0]["device_elapsed_ms"] = 0.01
        elif change == "total":
            value["workloads"][0]["iterations"][0]["host_forward_to_token_ns"] += 1
        else:
            value["protocol"]["sampling"] = "outside_timing"
        invalid(lambda: audit.validate_report(value, SPEC))


def wrong_kv_and_device_allocation_are_rejected():
    for kind in ("resident", "live_pages", "allocation", "cuda_in_cpu"):
        value = deepcopy(BASE)
        state = value["workloads"][0]["iterations"][1]["after_clear"]
        if kind == "resident":
            state["cpu"]["resident_kv_bytes"] = 0
        elif kind == "live_pages":
            state["cpu"]["used_kv_pages"] = 1
        elif kind == "allocation":
            state["allocation_stats"]["release_calls"] = 1
        else:
            state["cuda"] = {}
        invalid(lambda: audit.validate_report(value, SPEC))


def cuda_transfer_memory_and_weights_are_recomputed():
    result = audit.validate_report(GPU, SPEC)
    assert result["owned_device_bytes"] == 3449229312 and result["forwards"] == 585
    for field in ("weight_h2d_bytes", "metadata_h2d_bytes", "status_d2h_bytes", "debug_d2h_bytes",
                  "intermediate_h2d_bytes", "intermediate_d2h_bytes", "owned_device_bytes", "completed_forwards"):
        value = deepcopy(GPU)
        value["workloads"][0]["iterations"][0]["after_measured"]["cuda"][field] += 1
        invalid(lambda: audit.validate_report(value, SPEC))
    for change in ("alias", "shape", "plan", "dtype"):
        value = deepcopy(GPU)
        if change == "alias":
            value["runtime"]["weights"][1]["offset"] += 256
        elif change == "shape":
            value["runtime"]["weights"][2]["shape"][1] += 1
        elif change == "plan":
            value["initial_state"]["cuda"]["resident"]["total_owned_bytes"] += 1
        else:
            value["runtime"]["arithmetic"]["gemm_compute"] = "CUBLAS_COMPUTE_32F_FAST_TF32"
        invalid(lambda: audit.validate_report(value, SPEC))


def exact_noise_boundaries_and_negative_results():
    stats = audit.paired_statistics([100] * 5, [95, 95, 95, 95, 95])
    assert abs(stats["median_relative_percent"] + 5) < 1e-12
    assert audit.classify(stats, 5) == "inconclusive"
    slower = audit.paired_statistics([100] * 5, [120] * 5)
    assert audit.classify(slower, 5) == "slower"
    assert audit.classify(slower, 10) == "slower"
    assert audit.classify(slower, 10.0001) == "measurement_inconclusive"
    crossing = dict(slower, paired_ci95_percent=[-1, 25])
    assert audit.classify(crossing, 5) == "inconclusive"
    invalid(lambda: audit.paired_statistics([100] * 15, [90] * 15))
    invalid(lambda: audit.paired_statistics([True] * 5, [90] * 5))


def trial_medians_not_inner_repeats_are_statistical_units():
    reports = []
    for slot in audit.schedule():
        # 同一进程内的极端值保留，但不增加独立 trial 数。
        latency = 80 if slot["backend"] == "cuda" else 100
        report = dict(process=slot, mock=dict(cases={
            w["name"]: dict(samples=[latency, latency, 1000000], median_ns=latency, token_ids=[42])
            for w in SPEC["workloads"]}))
        reports.append(report)
    with patch.object(audit, "validate_report", side_effect=lambda r, _: r["mock"]):
        summary = audit.analyze_pairs(reports, SPEC)
        assert summary["independent_trials"] == 5 and summary["measured_repetitions"] == 2520
        for group in summary["comparisons"].values():
            for case in group.values():
                assert case["a_trial_median_ns"] == [100] * 5 and case["b_trial_median_ns"] == [80] * 5
                assert case["noise_percent"] == 5 and case["status"] == "faster"
        invalid(lambda: audit.analyze_pairs(reports[:-1], SPEC))
        changed = deepcopy(reports)
        changed[30]["process"] = changed[31]["process"]
        invalid(lambda: audit.analyze_pairs(changed, SPEC))


def cross_backend_generation_difference_is_not_hidden():
    reports = []
    for slot in audit.schedule():
        reports.append(dict(process=slot, mock=dict(cases={
            w["name"]: dict(samples=[100] * 3, median_ns=100, token_ids=[43 if slot["backend"] == "cuda" else 42])
            for w in SPEC["workloads"]})))
    with patch.object(audit, "validate_report", side_effect=lambda r, _: r["mock"]):
        summary = audit.analyze_pairs(reports, SPEC)
        assert summary["status"] == "correctness_followup_required"
        case = summary["comparisons"]["cpu8"]["generate-16-32"]
        assert not case["token_agreement"] and case["trial_tokens"][0] == dict(A=[42], B=[43])


def duplicates_and_path_escapes_are_rejected():
    invalid(lambda: audit.decode('{"status":"passed","status":"failed"}'))
    invalid(lambda: audit.decode('{"duration":NaN}'))
    for name in ("../x", "./x", "/x", "x//y", "C:/x", r"x\y"):
        invalid(lambda: audit.relative_path(name))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "data.json"
        path.write_text('{"value":1}', encoding="utf-8")
        assert audit.sha(path) == hashlib.sha256(path.read_bytes()).hexdigest()
        (Path(directory) / "link").symlink_to(path)
        invalid(lambda: audit.artifact(directory, "link"))


def preflight_and_publication_failure_preserve_previous_outputs():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        original = b'{"status":"old"}\n'
        (root / "summary.json").write_bytes(original)
        result = subprocess.run([sys.executable, str(ROOT / "scripts/analyze_cuda_benchmark.py"),
                                 "--directory", directory, "--write"], text=True, capture_output=True)
        assert result.returncode == 1 and "缺少证据" in result.stderr
        assert (root / "summary.json").read_bytes() == original
        replace, count = audit.os.replace, 0

        def fail_second(source, target):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("确定性发布故障")
            replace(source, target)

        try:
            with patch.object(audit.os, "replace", side_effect=fail_second):
                audit.publish(root, {"summary.json": dict(status="new"), "availability.json": {}})
        except OSError:
            pass
        else:
            raise AssertionError("发布故障未触发")
        assert (root / "summary.json").read_bytes() == original
        assert not (root / "availability.json").exists()
        assert list(root.iterdir()) == [root / "summary.json"]


def bundle_fixture(root):
    def write(name, value):
        path = root / name
        path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        return audit.sha(path)

    verify = ROOT / "scripts/analyze_cuda_benchmark.py"
    sources = {
        "apps/cuda_runtime_bench.cpp": b"fixture\n",
        "src/minillm/cuda/runtime.cpp": b"fixture\n",
        "scripts/analyze_cuda_benchmark.py": verify.read_bytes(),
        "benchmarks/runtime-inputs/qwen3-cuda-v0.json": (ROOT / "benchmarks/runtime-inputs/qwen3-cuda-v0.json").read_bytes(),
    }
    files = [dict(path=name, size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) for name, raw in sources.items()]
    scope = ["apps", "src", "scripts", "benchmarks/runtime-inputs"]
    source_hash = write("source-state.json", dict(scope=scope, files=files))
    write("numerical-source-state.json", dict(scope=scope, files=files))
    with zipfile.ZipFile(root / "source-snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sources.items():
            archive.writestr(name, raw)
    (root / "input.json").write_bytes(sources["benchmarks/runtime-inputs/qwen3-cuda-v0.json"])
    (root / "verify.py").write_bytes(verify.read_bytes())
    write("validation-summary.json", dict(status="passed", complete=True, passed=True, full_corpus_contract=True,
                                          model_sha256=audit.MODEL_SHA256, totals=dict(numeric_failures=0)))
    manifest = dict(schema_version=1, benchmark=audit.BENCHMARK, protocol_id="qwen3-cuda-model-v0", run_id="synthetic-fixture",
        reports=audit.schedule(), statistics=audit.STATISTICS, dependencies=dict(llama_commit=audit.LLAMA_COMMIT),
        build=dict(own_cuda="ON", upstream_cuda="OFF", type="RelWithDebInfo"),
        model=dict(sha256=audit.MODEL_SHA256), binary=dict(sha256="0" * 64), input=dict(path="input.json", sha256=audit.INPUT_SHA256),
        source=dict(state_file="source-state.json", worktree_state_sha256=source_hash, scope=scope,
                    snapshot=dict(path="source-snapshot.zip", sha256=audit.sha(root / "source-snapshot.zip"))),
        artifacts=[dict(path=name, sha256=audit.sha(root / name)) for name in
                   ("verify.py", "validation-summary.json", "numerical-source-state.json")])
    manifest_hash = write("manifest.json", manifest)
    identity = dict(run_id=manifest["run_id"], manifest_sha256=manifest_hash,
                    source_state_sha256=source_hash, binary_sha256="0" * 64)
    completed = []
    cpu16 = deepcopy(BASE)
    cpu16["backend"] = "cpu16"
    cpu16["runtime"]["configuration"]["threads"] = 16
    for slot in manifest["reports"]:
        report = dict({"cpu8": BASE, "cpu16": cpu16, "cuda": GPU}[slot["backend"]], process=slot,
                      scope="paired_model_baseline", run_identity=identity)
        record = dict(file=slot["file"], exit_code=0, sha256=write(slot["file"], report), artifacts=[])
        for suffix in (".stdout.txt", ".stderr.txt", ".process.json"):
            name = slot["file"] + suffix
            content = dict(order=slot["order"], exit_code=0, before={}, after={}, arguments=["fixture"])
            record["artifacts"].append(dict(path=name, sha256=write(name, content)))
        completed.append(record)
    write("collection-status.json", dict(status="passed", reports=completed))
    return manifest


def portable_full_bundle_revalidation_and_tamper_rejection():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bundle_fixture(root)
        result = audit.validate_bundle(root)
        assert result["summary"]["reports"] == 70 and result["summary"]["status"] == "measured"
        assert result["summary"]["measured_repetitions"] == 2520
        assert result["availability"]["status"] == "AVAILABLE"
        audit.publish(root, {"summary.json": result["summary"]})
        original = (root / "summary.json").read_bytes()
        first = root / audit.schedule()[0]["file"]
        raw = first.read_bytes()
        first.write_bytes(raw + b" ")
        invalid(lambda: audit.validate_bundle(root))
        assert (root / "summary.json").read_bytes() == original
        first.write_bytes(raw)
        (root / "source-snapshot.zip").unlink()
        invalid(lambda: audit.validate_bundle(root))
        assert (root / "summary.json").read_bytes() == original


def executable_preflight_guards(executable):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        existing = root / "existing.json"
        existing.write_bytes(b"original\n")
        help_result = subprocess.run([executable, "--help"], text=True, capture_output=True)
        assert help_result.returncode == 0 and "cpu8|cpu16|cuda" in help_result.stdout
        result = subprocess.run([executable, "--output", str(existing)], text=True, capture_output=True)
        assert result.returncode == 1 and existing.read_bytes() == b"original\n"
        model = root / "not-model.gguf"
        model.write_bytes(b"not a model\n")
        recipe = root / "input.json"
        recipe.write_bytes((ROOT / "benchmarks/runtime-inputs/qwen3-cuda-v0.json").read_bytes())
        for name, backend, change in (("backend", "upstream", False), ("model", "cuda", False), ("input", "cpu8", True)):
            if change:
                value = deepcopy(SPEC)
                value["workloads"].pop()
                recipe.write_text(json.dumps(value), encoding="utf-8")
            report = root / f"{name}.json"
            result = subprocess.run([executable, "--model", str(model), "--input", str(recipe),
                                     "--output", str(report), "--backend", backend], text=True, capture_output=True)
            assert result.returncode == 1 and audit.read(report)["status"] == "failed"
            assert {"backend": "--backend", "model": "模型摘要", "input": "冻结"}[name] in result.stderr
        print("[PASS] executable_preflight_guards")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable")
    options = parser.parse_args()
    tests = [frozen_schedule_is_separate_balanced_and_complete, complete_workload_report_has_exact_counts,
             missing_warmup_setup_and_cases_are_rejected, changed_inputs_and_natural_history_are_rejected,
             no_boolean_nonfinite_or_excluded_sampling_time, wrong_kv_and_device_allocation_are_rejected,
             cuda_transfer_memory_and_weights_are_recomputed,
             exact_noise_boundaries_and_negative_results, trial_medians_not_inner_repeats_are_statistical_units,
             cross_backend_generation_difference_is_not_hidden, duplicates_and_path_escapes_are_rejected,
             preflight_and_publication_failure_preserve_previous_outputs,
             portable_full_bundle_revalidation_and_tamper_rejection]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    if options.executable:
        executable_preflight_guards(options.executable)
    count = len(tests) + bool(options.executable)
    print(f"{count}/{count} tests passed")
