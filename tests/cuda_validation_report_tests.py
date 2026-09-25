"""CUDA 数值证据的确定性反例；无需模型权重或 CUDA 设备。"""

from copy import deepcopy
import hashlib
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_cuda_validation as audit

CONTRACT = audit.read(ROOT / "tests/data/qwen3_validation_cases.json")


def invalid(operation):
    try:
        operation()
    except audit.ValidationError:
        return
    raise AssertionError("无效证据未被拒绝")


def score(token=42, margin=0.01, label="actual"):
    return dict(token=token, margin=margin, sha256=hashlib.sha256(label.encode()).hexdigest())


def comparison(actual=None, expected=None, **values):
    actual = actual or score()
    expected = expected or score(label="reference")
    result = dict(rmse=0.01, max_absolute=0.04, cosine=0.99999, all_finite=True,
                  actual_sha256=actual["sha256"], reference_sha256=expected["sha256"],
                  actual_argmax=actual["token"], reference_argmax=expected["token"],
                  reference_margin=expected["margin"], reference="cpu")
    result.update(values)
    result["near_tie"] = result["reference_margin"] <= 2 * result["max_absolute"]
    result["argmax_equal"] = result["actual_argmax"] == result["reference_argmax"]
    result["passed"] = (result["rmse"] < 0.05 and result["max_absolute"] < 0.5
                        and result["cosine"] >= 0.9999 and (result["near_tie"] or result["argmax_equal"]))
    return result


def full_case_matrix_and_input_hash():
    plan = audit.case_plan(CONTRACT)
    assert len(plan) == len({c["id"] for c in plan}) == 240
    assert sum(c["sampled_rows"] for c in plan) == 3920
    groups = audit.batches(CONTRACT, 0, 33, 33, 4)
    assert len(groups) == 4 and groups[1][0][1:3] == (8, 1)
    raw = b"".join(struct.pack("<i", len(group)) + b"".join(struct.pack("<4i", *row) for row in group)
                   for group in groups)
    assert audit.input_digest(groups) == hashlib.sha256(raw).hexdigest()
    assert audit.input_digest(groups) != audit.input_digest(audit.batches(CONTRACT, 0, 33, 16, 4))


def exact_threshold_boundaries():
    for values, expected in (({}, True), (dict(rmse=0.05, max_absolute=0.05), False),
                             (dict(max_absolute=0.5), False), (dict(cosine=0.9999), True),
                             (dict(cosine=0.999899), False)):
        record = comparison(**values)
        assert audit.compare_record(record, CONTRACT["thresholds"]) is expected
        record["passed"] = not record["passed"]
        invalid(lambda: audit.compare_record(record, CONTRACT["thresholds"]))


def reference_attention_precision_is_explicit():
    settings = dict(flash_attention=False, kv_dtype="F16", threads=8,
                    qk_accumulation_dtype="F32", pv_accumulation_dtype="F32")
    report = dict(references=dict(llama_f32_settings=settings))
    summary = dict(reference_attention="unfused")
    audit.reference_settings(report, summary)
    for field, value in (("flash_attention", True), ("kv_dtype", "F32"),
                         ("pv_accumulation_dtype", "F16"), ("threads", 16)):
        changed = deepcopy(report)
        changed["references"]["llama_f32_settings"][field] = value
        invalid(lambda: audit.reference_settings(changed, summary))


def near_tie_formula_is_not_a_mismatch_override():
    actual, expected = score(token=43), score(margin=0.01, label="ref")
    record = comparison(actual, expected)
    assert audit.compare_record(record, CONTRACT["thresholds"], expected, actual)
    assert record["near_tie"] and not record["argmax_equal"]
    record = comparison(actual, score(margin=0.081, label="ref"))
    assert not audit.compare_record(record, CONTRACT["thresholds"])
    record["near_tie"] = record["passed"] = True
    invalid(lambda: audit.compare_record(record, CONTRACT["thresholds"]))


def score_identity_and_margin_must_match():
    actual, expected = score(), score(label="ref")
    record = comparison(actual, expected)
    assert audit.compare_record(record, CONTRACT["thresholds"], expected, actual)
    for field, value in (("reference_sha256", "0" * 64), ("actual_sha256", "0" * 64),
                         ("reference_argmax", 43), ("actual_argmax", 43), ("reference_margin", 0.02)):
        broken = deepcopy(record)
        broken[field] = value
        invalid(lambda: audit.compare_record(broken, CONTRACT["thresholds"], expected, actual))


def nonfinite_and_boolean_metrics_are_rejected():
    for field in ("rmse", "max_absolute", "cosine", "reference_margin"):
        for value in (float("nan"), float("inf"), True):
            record = comparison()
            record[field] = value
            invalid(lambda: audit.compare_record(record, CONTRACT["thresholds"]))
    for value in (score(token=-1), score(token=151936), score(token=True), score(margin=float("inf"))):
        invalid(lambda: audit.check_score(value, 151936))


def inconsistent_extrema_and_status_are_rejected():
    for values in (dict(rmse=0.041), dict(cosine=1.01), dict(reference_margin=-0.01),
                   dict(actual_argmax=-1)):
        invalid(lambda: audit.compare_record(comparison(**values), CONTRACT["thresholds"]))
    record = comparison()
    record["argmax_equal"] = False
    invalid(lambda: audit.compare_record(record, CONTRACT["thresholds"]))


def generation_fixture():
    identity, length = "zh-l16-g32", 16
    references = {name: [score(label=f"{name}-{i}") for i in range(32)] for name in ("cpu", "llama_f32")}
    first, rows = dict(cpu=None, llama_f32=None), []
    for step in range(32):
        actual = score(token=43 if step == 1 else 42, label=f"gpu-{step}")
        row = dict(step=step, cuda=actual, references={})
        for name in references:
            record = None
            if step <= 1:
                record = comparison(actual, references[name][step], case=identity, reference=name,
                                    step=step, sequence=0, position=length - 1 + step)
            if step == 1:
                first[name] = record
            row["references"][name] = dict(score=references[name][step], input_prefix_equal=step <= 1,
                                            comparison=record)
        rows.append(row)
    return dict(id=identity, prompt_tokens=length, rows=rows, first_divergence=first, passed=True), references


def natural_generation_keeps_only_common_input_comparisons():
    report, references = generation_fixture()
    metrics = audit.Metrics()
    first, passed = audit.generation_rows(report, references, CONTRACT, metrics)
    assert passed and first["cpu"]["step"] == first["llama_f32"]["step"] == 1
    assert metrics.totals["generation_comparisons"] == 4
    assert metrics.totals["argmax_divergences"] == 2
    assert metrics.totals["numeric_failures"] == 0


def drifted_inputs_cannot_claim_comparability():
    for change in ("restored_prefix", "extra_comparison", "missing_comparison", "missing_first", "wrong_score"):
        report, references = generation_fixture()
        if change == "restored_prefix":
            report["rows"][2]["references"]["cpu"]["input_prefix_equal"] = True
        elif change == "extra_comparison":
            report["rows"][2]["references"]["cpu"]["comparison"] = comparison()
        elif change == "missing_comparison":
            report["rows"][0]["references"]["cpu"]["comparison"] = None
        elif change == "missing_first":
            report["first_divergence"]["cpu"] = None
        else:
            report["rows"][0]["references"]["cpu"]["score"] = score(token=90)
        invalid(lambda: audit.generation_rows(report, references, CONTRACT, audit.Metrics()))


def transfer_fixture():
    baseline = dict(resident={}, weight_h2d_bytes=100, rope_h2d_bytes=32, owned_device_allocations=4,
                    owned_device_bytes=1024, model_load_ns=1, storage_initialization_ns=2, weight_decode_upload_ns=1)
    value = dict(baseline, state="ready", post_launch_failures=0, sequence_lengths=[33, 0, 0, 0],
                 live_kv_tokens=33, live_sequences=1, kv_capacity_tokens=8192,
                 metadata_h2d_bytes=33 * 12 + 6 * 4, token_d2h_bytes=6 * 4, status_d2h_bytes=3 * 8,
                 debug_d2h_bytes=6 * 4 * 151936, completed_forwards=3,
                 intermediate_h2d_bytes=0, intermediate_d2h_bytes=0)
    transfer = audit.Transfers()
    transfer.add(33, 6, 3)
    return baseline, value, transfer


def copy_counts_and_state_are_recomputed():
    baseline, value, transfer = transfer_fixture()
    transfer.check(value, baseline, [33, 0, 0, 0], 151936)
    for name in ("metadata_h2d_bytes", "token_d2h_bytes", "status_d2h_bytes", "debug_d2h_bytes",
                 "weight_h2d_bytes", "completed_forwards", "intermediate_d2h_bytes", "post_launch_failures"):
        changed = dict(value)
        changed[name] += 1
        invalid(lambda: transfer.check(changed, baseline, [33, 0, 0, 0], 151936))
    changed = dict(value, state="poisoned")
    invalid(lambda: transfer.check(changed, baseline, [33, 0, 0, 0], 151936))
    invalid(lambda: transfer.check(value, baseline, [32, 0, 0, 0], 151936))


def artifact_paths_and_duplicate_json_are_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        good = root / "good.json"
        good.write_text('{"status":"passed"}', encoding="utf-8")
        assert audit.artifact(root, "good.json") == good
        for path in ("../good.json", "./good.json", "/good.json", "C:/good.json", "x//good.json", r"x\good.json", "missing"):
            invalid(lambda: audit.artifact(root, path))
        bad = root / "duplicate.json"
        bad.write_text('{"status":"passed","status":"failed"}', encoding="utf-8")
        invalid(lambda: audit.read(bad))


def missing_report_does_not_overwrite_existing_evidence():
    with tempfile.TemporaryDirectory() as directory:
        old = Path(directory) / "validation-summary.json"
        original = b'{"status":"passed","previous":true}\n'
        old.write_bytes(original)
        result = subprocess.run([sys.executable, str(ROOT / "scripts/analyze_cuda_validation.py"),
                                 "--directory", directory], capture_output=True, text=True)
        assert result.returncode == 1 and "缺少证据" in result.stderr
        assert old.read_bytes() == original


def negative_metrics_remain_in_summary():
    metrics = audit.Metrics()
    success = comparison(reference="cuda_canonical")
    failure = comparison(max_absolute=0.5, reference="cpu")
    metrics.add(success, True)
    metrics.add(failure, False)
    assert metrics.totals["teacher_comparisons"] == metrics.totals["generation_comparisons"] == 1
    assert metrics.totals["numeric_failures"] == 1 and metrics.first_failure == failure
    assert metrics.extrema["absolute_max"] == 0.5
    assert metrics.by_reference["cpu"]["numeric_failures"] == 1


if __name__ == "__main__":
    tests = [full_case_matrix_and_input_hash, exact_threshold_boundaries, reference_attention_precision_is_explicit,
             near_tie_formula_is_not_a_mismatch_override, score_identity_and_margin_must_match,
             nonfinite_and_boolean_metrics_are_rejected, inconsistent_extrema_and_status_are_rejected,
             natural_generation_keeps_only_common_input_comparisons, drifted_inputs_cannot_claim_comparability,
             copy_counts_and_state_are_recomputed, artifact_paths_and_duplicate_json_are_rejected,
             missing_report_does_not_overwrite_existing_evidence, negative_metrics_remain_in_summary]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"{len(tests)}/{len(tests)} tests passed")
