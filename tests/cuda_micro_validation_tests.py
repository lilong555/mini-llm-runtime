"""微基准形状、样本、统计、身份与归档反例；不需要模型或 CUDA 设备。"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_cuda_micro as audit
from cuda_benchmark_validation_tests import GPU

SPEC_PATH = ROOT / "benchmarks/runtime-inputs/qwen3-cuda-micro-v0.json"
SPEC = audit.read(SPEC_PATH)


def invalid(function):
    try:
        function()
    except audit.ValidationError:
        return
    raise AssertionError("无效 micro 证据未被拒绝")


def fixture():
    report = dict(schema_version=1, benchmark=audit.BENCHMARK, status="passed", trial=0,
                  input_sha256=audit.INPUT_SHA256, model_sha256=audit.MODEL_SHA256,
                  protocol=deepcopy(SPEC["measurement"]), dimensions=deepcopy(audit.DIMENSIONS),
                  arithmetic=deepcopy(audit.ARITHMETIC), device=deepcopy(GPU["runtime"]["device"]),
                  weights=deepcopy(GPU["runtime"]["weights"]), run_identity=None, cases=[],
                  model_load_ns=1000, storage_initialization_ns=2000, weight_decode_upload_ns=1500,
                  kv_initialization_ns=3000, rope_h2d_bytes=2048 * 128 * 4,
                  kv_initialization_transfers=audit.transfers(h2d_bytes=67174400, d2h_bytes=8, h2d_calls=256, d2h_calls=1))
    plan, payload, _ = audit.common.memory_plan(report)
    report.update(memory_plan=plan, weight_h2d_bytes=payload, owned_device_bytes=plan["total_owned_bytes"])
    steady = dict(audit.ZERO_ALLOCATIONS, allocation_calls=4, allocations=4, allocated_bytes=plan["total_owned_bytes"])
    report["before_initialization_allocations"] = deepcopy(audit.ZERO_ALLOCATIONS)
    report["before_cases_allocations"] = deepcopy(steady)
    report["after_cases_allocations"] = deepcopy(steady)
    report["after_destruction_allocations"] = dict(steady, release_calls=4, releases=4)
    for case in audit.make_cases(SPEC, audit.DIMENSIONS):
        indices = audit.point_indices(case)
        digest = hashlib.sha256(case["name"].encode()).hexdigest()
        verification = dict(all_finite=True, checked_elements=len(indices) if indices else case["output_elements"],
                            max_absolute=0.0, max_tolerance_ratio=0.0, output_sha256=digest,
                            points=[dict(index=i, actual=0.0, reference=0.0) for i in indices],
                            reference_sha256=None if indices else digest)
        copy = audit.expected_transfers(case)
        actual = dict(deepcopy(case), status="passed", input_sha256=audit.case_input_hash(case),
                      before_allocations=deepcopy(steady), after_allocations=deepcopy(steady),
                      preparation_host_ns=10, preparation_transfers=copy["preparation"], samples=[])
        for iteration in range(5):
            actual["samples"].append(dict(iteration=iteration, phase="warmup" if iteration < 2 else "measured",
                calls=32, setup_host_ns=10, host_enqueue_to_completion_ns=320000, device_interval_ms=0.3,
                validation_host_ns=100, verification=deepcopy(verification),
                **{phase + "_transfers": deepcopy(copy[phase]) for phase in ("setup", "measured", "validation")}))
        report["cases"].append(actual)
    return report


BASE = fixture()


def reports():
    result = []
    for trial in range(5):
        report = deepcopy(BASE)
        report["trial"] = trial
        if trial % 2:
            report["cases"].reverse()
        result.append(report)
    return result


def frozen_shapes_inputs_and_sample_counts():
    assert audit.sha(SPEC_PATH) == audit.INPUT_SHA256
    planned = audit.make_cases(SPEC, audit.DIMENSIONS)
    assert Counter(v["operation"] for v in planned) == dict(matrix=72, rms_norm=27, rope=108, softmax=84, attention=84)
    assert [v["input_seed"] for v in planned] == list(range(1, 376))
    result = audit.validate_report(BASE, SPEC)
    assert result["samples"] == 1875 and result["measured_samples"] == 1125 and result["api_calls"] == 60000
    assert result["owned_device_bytes"] == 3449229312
    assert audit.case_input_hash(planned[0]) == "09c7f6ac7e479ced9d955b11aef2b9749bad078b8f5e084b7098f69be4f233de"


def missing_samples_cases_and_wrong_order_are_rejected():
    for change in ("case", "sample", "phase", "calls", "order", "trial"):
        report = deepcopy(BASE)
        if change == "case":
            report["cases"].pop()
        elif change == "sample":
            report["cases"][0]["samples"].pop(0)
        elif change == "phase":
            report["cases"][0]["samples"][0]["phase"] = "measured"
        elif change == "calls":
            report["cases"][0]["samples"][0]["calls"] = 16
        elif change == "order":
            report["cases"].reverse()
        else:
            report["trial"] = True
        invalid(lambda: audit.validate_report(report, SPEC))
    reverse = deepcopy(BASE)
    reverse["trial"] = 1
    invalid(lambda: audit.validate_report(reverse, SPEC))
    reverse["cases"].reverse()
    audit.validate_report(reverse, SPEC)


def dimensions_recipe_weights_and_causal_metadata_are_checked():
    for change in ("dimension", "shape", "seed", "input", "slot", "position", "weight", "arithmetic"):
        report = deepcopy(BASE)
        if change == "dimension":
            report["dimensions"]["vocabulary"] -= 1
        elif change == "shape":
            report["cases"][0]["n"] += 1
        elif change == "seed":
            report["cases"][0]["input_seed"] += 1
        elif change == "input":
            report["cases"][0]["input_sha256"] = "0" * 64
        elif change in ("slot", "position"):
            report["cases"][-1]["slots" if change == "slot" else "positions"][-1] += 1
        elif change == "weight":
            report["weights"][1]["offset"] += 256
        else:
            report["arithmetic"]["gemm_compute"] = "CUBLAS_COMPUTE_32F_FAST_TF32"
        invalid(lambda: audit.validate_report(report, SPEC))


def timing_and_copy_boundaries_cannot_be_changed():
    for key in ("host_enqueue_to_completion_ns", "device_interval_ms", "setup_host_ns", "validation_host_ns"):
        for bad in (True, 0, -1, float("nan"), float("inf")):
            report = deepcopy(BASE)
            report["cases"][0]["samples"][0][key] = bad
            invalid(lambda: audit.validate_report(report, SPEC))
    for change in ("setup_transfers", "measured_transfers", "validation_transfers", "preparation", "kv", "protocol"):
        report = deepcopy(BASE)
        if change == "preparation":
            report["cases"][0]["preparation_transfers"]["d2h_bytes"] += 4
        elif change == "kv":
            report["kv_initialization_transfers"]["h2d_calls"] -= 1
        elif change == "protocol":
            report["protocol"]["status_and_output_download"] = "inside_timing"
        else:
            report["cases"][0]["samples"][0][change]["h2d_bytes"] += 4
        invalid(lambda: audit.validate_report(report, SPEC))


def numerical_points_metrics_and_full_output_contract_are_checked():
    for change in ("finite", "count", "index", "actual", "summary", "limit", "reference"):
        report = deepcopy(BASE)
        value = report["cases"][72 if change == "reference" else 0]["samples"][0]["verification"]
        if change == "finite":
            value["all_finite"] = 1
        elif change == "count":
            value["checked_elements"] -= 1
        elif change == "index":
            value["points"][0]["index"] += 1
        elif change == "actual":
            value["points"][0]["actual"] = 1.0
        elif change == "summary":
            value["max_absolute"] = 0.1
        elif change == "limit":
            value["max_tolerance_ratio"] = 1.00001
        else:
            value["reference_sha256"] = None
        invalid(lambda: audit.validate_report(report, SPEC))
    report = deepcopy(BASE)
    report["cases"][0]["samples"][1]["verification"]["output_sha256"] = "0" * 64
    invalid(lambda: audit.validate_report(report, SPEC))


def allocation_release_and_memory_plan_are_exact():
    for change in ("before", "after", "destruction", "memory", "upload"):
        report = deepcopy(BASE)
        if change in ("before", "after"):
            report["cases"][0][change + "_allocations"]["allocation_calls"] += 1
        elif change == "destruction":
            report["after_destruction_allocations"]["releases"] -= 1
        elif change == "memory":
            report["memory_plan"]["total_owned_bytes"] += 1
        else:
            report["weight_h2d_bytes"] += 4
        invalid(lambda: audit.validate_report(report, SPEC))


def independent_trials_not_calls_or_inner_repeats():
    value = audit.trial_statistics([[1, 2, 1e9]] * 5)
    assert value["trial_median_ns_per_call"] == [2] * 5
    assert value["median_ns_per_call"] == 2 and value["median_ci95_ns_per_call"] == [2, 2]
    assert value["samples_ns_per_call"][0][-1] == 1e9
    invalid(lambda: audit.trial_statistics([[2] * 3] * 15))
    invalid(lambda: audit.trial_statistics([[2] * 32] * 5))
    invalid(lambda: audit.trial_statistics([[True] * 3] * 5))
    sequence = reports()
    invalid(lambda: audit.summarize(sequence[:-1], SPEC))
    sequence[-1]["trial"] = 3
    invalid(lambda: audit.summarize(sequence, SPEC))
    sequence = reports()
    for sample in sequence[-1]["cases"][0]["samples"]:
        sample["verification"]["output_sha256"] = "0" * 64
    invalid(lambda: audit.summarize(sequence, SPEC))


def bundle_fixture(root):
    def write(name, value):
        (root / name).write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        return audit.sha(root / name)

    sources = {name: b"fixture\n" for name in ("apps/cuda_runtime_bench.cpp", "src/minillm/cuda/runtime.cpp",
                                             "apps/cuda_kernel_bench.cpp", "apps/cuda_micro_protocol.h",
                                             "src/minillm/cuda/attention.cu")}
    for name in ("scripts/analyze_cuda_micro.py", "scripts/analyze_cuda_benchmark.py",
                 "benchmarks/runtime-inputs/qwen3-cuda-v0.json", "benchmarks/runtime-inputs/qwen3-cuda-micro-v0.json"):
        sources[name] = (ROOT / name).read_bytes()
    scope = ["apps", "src", "scripts", "benchmarks/runtime-inputs"]
    files = [dict(path=name, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)) for name, raw in sources.items()]
    state_hash = write("source-state.json", dict(scope=scope, files=files))
    with zipfile.ZipFile(root / "source-snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sources.items():
            archive.writestr(name, raw)
    shutil.copyfile(SPEC_PATH, root / "input.json")
    shutil.copyfile(ROOT / "scripts/analyze_cuda_micro.py", root / "verify.py")
    shutil.copyfile(ROOT / "scripts/analyze_cuda_benchmark.py", root / "analyze_cuda_benchmark.py")
    manifest = dict(schema_version=1, benchmark=audit.BENCHMARK, protocol_id=audit.PROTOCOL_ID, run_id="synthetic-fixture",
        reports=audit.schedule(), statistics=audit.STATISTICS, dependencies=dict(llama_commit=audit.common.LLAMA_COMMIT),
        build=dict(own_cuda="ON", upstream_cuda="OFF", type="RelWithDebInfo"),
        model=dict(sha256=audit.MODEL_SHA256), binary=dict(sha256="0" * 64), input=dict(path="input.json", sha256=audit.INPUT_SHA256),
        source=dict(scope=scope, state_file="source-state.json", worktree_state_sha256=state_hash,
                    snapshot=dict(path="source-snapshot.zip", sha256=audit.sha(root / "source-snapshot.zip"))),
        artifacts=[dict(path=name, sha256=audit.sha(root / name)) for name in ("verify.py", "analyze_cuda_benchmark.py")])
    manifest_hash = write("manifest.json", manifest)
    identity = dict(run_id=manifest["run_id"], manifest_sha256=manifest_hash, source_state_sha256=state_hash, binary_sha256="0" * 64)
    completed = []
    for slot, report in zip(audit.schedule(), reports()):
        report["run_identity"] = identity
        record = dict(file=slot["file"], exit_code=0, sha256=write(slot["file"], report), artifacts=[])
        for suffix in (".stdout.txt", ".stderr.txt", ".process.json"):
            name = slot["file"] + suffix
            content = dict(trial=slot["trial"], exit_code=0, before={}, after={}, arguments=["fixture"])
            record["artifacts"].append(dict(path=name, sha256=write(name, content)))
        completed.append(record)
    write("collection-status.json", dict(status="passed", planned_reports=5, reports=completed))


def portable_bundle_and_missing_tampered_artifacts():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        bundle = root / "bundle"
        bundle.mkdir()
        bundle_fixture(bundle)
        result = audit.validate_bundle(bundle)
        assert result["summary"]["measured_samples"] == 5625 and result["summary"]["case_count"] == 375
        assert result["summary"]["api_calls"] == 300000 and result["summary"]["end_to_end_speedup"] is None
        audit.publish(bundle, {"summary.json": result["summary"]})
        moved = root / "moved"
        shutil.copytree(bundle, moved)
        process = subprocess.run([sys.executable, str(moved / "verify.py"), "--directory", str(moved)],
                                 cwd=root, capture_output=True, text=True)
        assert process.returncode == 0, process.stderr
        saved = (bundle / "summary.json").read_bytes()
        for name in ("source-snapshot.zip", "cuda-micro-t0.json", "cuda-micro-t0.json.stderr.txt", "analyze_cuda_benchmark.py"):
            path = bundle / name
            raw = path.read_bytes()
            path.unlink()
            invalid(lambda: audit.validate_bundle(bundle))
            path.write_bytes(raw + b" ")
            invalid(lambda: audit.validate_bundle(bundle))
            path.write_bytes(raw)
            assert (bundle / "summary.json").read_bytes() == saved


def publication_failure_preserves_existing_summary():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        summary = root / "summary.json"
        summary.write_bytes(b"original\n")
        result = subprocess.run([sys.executable, str(ROOT / "scripts/analyze_cuda_micro.py"),
                                 "--directory", directory, "--write"], capture_output=True, text=True)
        assert result.returncode == 1 and summary.read_bytes() == b"original\n"
        original_replace = audit.common.os.replace
        calls = 0

        def fail_second(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("确定性发布故障")
            original_replace(source, target)

        try:
            with patch.object(audit.common.os, "replace", side_effect=fail_second):
                audit.publish(root, {"summary.json": {}, "availability.json": {}})
        except OSError:
            pass
        else:
            raise AssertionError("发布故障没有触发")
        assert summary.read_bytes() == b"original\n" and not (root / "availability.json").exists()


def executable_guards(executable):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        existing = root / "existing.json"
        existing.write_bytes(b"original\n")
        help_result = subprocess.run([executable, "--help"], capture_output=True, text=True)
        assert help_result.returncode == 0 and "mini-cuda-kernel-bench" in help_result.stdout
        result = subprocess.run([executable, "--output", str(existing)], capture_output=True, text=True)
        assert result.returncode == 1 and existing.read_bytes() == b"original\n"
        model = root / "not-model.gguf"
        model.write_bytes(b"not a model\n")
        recipe = root / "input.json"
        for changed in (False, True):
            recipe.write_bytes(SPEC_PATH.read_bytes() + (b" " if changed else b""))
            output = root / f"rejected-{changed}.json"
            result = subprocess.run([executable, "--model", str(model), "--input", str(recipe), "--output", str(output)],
                                    capture_output=True, text=True)
            assert result.returncode == 1 and audit.read(output)["status"] == "failed"
            assert ("冻结" if changed else "模型") in result.stderr
        print("[PASS] executable_guards")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable")
    options = parser.parse_args()
    tests = [frozen_shapes_inputs_and_sample_counts, missing_samples_cases_and_wrong_order_are_rejected,
             dimensions_recipe_weights_and_causal_metadata_are_checked, timing_and_copy_boundaries_cannot_be_changed,
             numerical_points_metrics_and_full_output_contract_are_checked, allocation_release_and_memory_plan_are_exact,
             independent_trials_not_calls_or_inner_repeats, portable_bundle_and_missing_tampered_artifacts,
             publication_failure_preserves_existing_summary]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    if options.executable:
        executable_guards(options.executable)
    count = len(tests) + bool(options.executable)
    print(f"{count}/{count} tests passed")
