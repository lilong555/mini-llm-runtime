"""严格复核真实形状 CUDA 微基准；独立 trial 统计，不推导模型加速或硬件带宽。"""

import argparse
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import struct
import sys
import zipfile

import analyze_cuda_benchmark as common
from analyze_cuda_benchmark import (ValidationError, artifact, digest, finite, identical, integer,
                                    publish, read, require, sha)

BENCHMARK = "minillm-cuda-micro"
PROTOCOL_ID = "qwen3-cuda-micro-v0"
INPUT_SHA256 = "6aa2bc8af5de001ab3a8baedd305d0c77822a48b5baddc8f5708e79f496e706c"
MODEL_SHA256 = common.MODEL_SHA256
DIMENSIONS = dict(embedding=1024, layers=28, heads=16, kv_heads=8, head_dim=128, feed_forward=3072,
                  vocabulary=151936, rms_epsilon=struct.unpack("<f", struct.pack("<f", 1e-6))[0],
                  rope_base=1000000.0)
ARITHMETIC = dict(source_weight_dtype="Q8_0", device_weight_dtype="F32", activation_dtype="F32",
                  kv_dtype="F16", kv_rounding="nearest_even", qk_pv_accumulation_dtype="F32",
                  softmax_exponential_dtype="F32", softmax_denominator_dtype="F64",
                  gemm_compute="CUBLAS_COMPUTE_32F_PEDANTIC", fast_math=False)
STATISTICS = dict(independent_trials=5, process_median_repeats=3, calls_per_sample=32,
                  statistics_unit="independent_trial_median",
                  confidence_interval="exact_percentile_bootstrap_median_95", bootstrap_resamples=3125,
                  comparison=False, hardware_dram_bandwidth=None)
ZERO_ALLOCATIONS = dict(allocation_calls=0, allocations=0, release_calls=0, releases=0, allocated_bytes=0)


def schedule():
    return [dict(trial=i, file=f"cuda-micro-t{i}.json", order="reverse" if i % 2 else "canonical") for i in range(5)]


def sample_rows(count):
    return sorted({0, count // 2, count - 1})


def sample_columns(count):
    return list(range(count)) if count <= 8 else [i * (count - 1) // 7 for i in range(8)]


def make_cases(spec, d):
    result = []
    q, kv = d["heads"] * d["head_dim"], d["kv_heads"] * d["head_dim"]
    rows = spec["matrix"]["rows"]

    def add(name, op, role, m, n, k=0, tensor="", group=0, context=0, slots=(), positions=()):
        result.append(dict(name=name, operation=op, role=role, tensor=tensor, m=m, n=n, k=k,
            group_width=group, max_context=context, slots=list(slots), positions=list(positions),
            input_seed=len(result) + 1, logical_flops_per_call=2 * m * n * k if op == "matrix" else None,
            output_elements=m * n, query_heads=d["heads"], kv_heads=d["kv_heads"], head_dim=d["head_dim"],
            kv_max_length=2048))

    matrices = dict(Q=("attn_q", q, d["embedding"]), K=("attn_k", kv, d["embedding"]),
        V=("attn_v", kv, d["embedding"]), attention_output=("attn_output", d["embedding"], q),
        gate=("ffn_gate", d["feed_forward"], d["embedding"]), up=("ffn_up", d["feed_forward"], d["embedding"]),
        down=("ffn_down", d["embedding"], d["feed_forward"]), LM_head=("output", d["vocabulary"], d["embedding"]))
    for role in spec["matrix"]["roles"]:
        tensor, n, k = matrices[role]
        tensor = ("" if role == "LM_head" else "blk.0.") + tensor + ".weight"
        for m in rows:
            add(f"matrix-{role}-m{m}", "matrix", role, m, n, k, tensor)
    for role in spec["ops"]["rms_norm_roles"]:
        n, group, tensor = (d["embedding"], d["embedding"], "attn_norm") if role == "hidden" else (
            q if role == "query" else kv, d["head_dim"], "attn_q_norm" if role == "query" else "attn_k_norm")
        for m in rows:
            add(f"rms_norm-{role}-m{m}", "rms_norm", role, m, n, tensor=f"blk.0.{tensor}.weight", group=group)
    for role in spec["ops"]["rope_roles"]:
        for m in rows:
            for position in spec["ops"]["rope_positions"]:
                add(f"rope-{role}-m{m}-p{position}", "rope", role, m, q if role == "query" else kv,
                    group=d["head_dim"], positions=[position] * m)

    def attention(role, slots, positions):
        m, length = len(slots), max(positions) + 1
        for op in ("softmax", "attention"):
            add(f"{op}-{role}-m{m}-l{length}", op, role, m, d["heads"] * 2048 if op == "softmax" else q,
                group=d["head_dim"], context=length, slots=slots, positions=positions)

    for length in spec["ops"]["attention_contexts"]:
        for m in spec["ops"]["decode_sequences"]:
            attention("decode", list(range(m)), [length - 1] * m)
        for m in rows:
            if m <= length:
                attention("prefill", [0] * m, list(range(length - m, length)))
    for prefix in spec["ops"]["mixed"]["prefix_tokens"]:
        attention("mixed", [0] * 16 + [1, 2], list(range(16)) + [prefix, prefix])
    require(len(result) == 375 and len({v["name"] for v in result}) == 375, "micro 用例集合无效")
    return result


def point_indices(case):
    m, n, op = case["m"], case["n"], case["operation"]
    if op == "matrix":
        return [r * n + c for c in sample_columns(n) for r in sample_rows(m)]
    if op == "attention":
        return [r * n + h * 128 + c for r in sample_rows(m) for h in (0, 7, 8, 15) for c in (0, 1, 63, 127)]
    return []


@lru_cache(maxsize=512)
def input_hash(operation, m, n, k, seed, positions):
    state = hashlib.sha256()
    if operation == "softmax":
        tail = struct.pack("<I", 0x7fc00000) * 2048
        for row, last in enumerate(positions):
            length = last + 1
            for head in range(16):
                cycle = struct.pack("<127f", *[((p * 7 + head * 3 + row * 11) % 127 - 63) / 16 for p in range(127)])
                state.update(cycle * (length // 127) + cycle[:(length % 127) * 4])
                state.update(tail[:(2048 - length) * 4])
    else:
        count = m * (k if operation == "matrix" else n)
        cycle = struct.pack("<127f", *[((i * 17 + seed * 13) % 127 - 63) / 4096 for i in range(127)])
        state.update(cycle * (count // 127) + cycle[:(count % 127) * 4])
    return state.hexdigest()


def case_input_hash(case):
    return input_hash(*(case[key] for key in ("operation", "m", "n", "k", "input_seed")), tuple(case["positions"]))


def transfers(h2d_bytes=0, d2h_bytes=0, h2d_calls=0, d2h_calls=0):
    return dict(h2d_bytes=h2d_bytes, d2h_bytes=d2h_bytes, h2d_calls=h2d_calls, d2h_calls=d2h_calls)


def expected_transfers(case):
    op, m, n, k = (case[key] for key in ("operation", "m", "n", "k"))
    upload = 4 * m * (k if op == "matrix" else n)
    metadata = len(case["slots"]) + len(case["positions"])
    reset = m * n if op == "softmax" else 0
    return dict(
        preparation=transfers(d2h_bytes=2048 * 128 * 4, d2h_calls=1) if op == "rope" else transfers(),
        setup=transfers(h2d_bytes=upload + 4 * (metadata + reset),
            h2d_calls=1 + bool(case["slots"]) + bool(case["positions"]) + bool(reset)),
        measured=transfers(), validation=transfers(d2h_bytes=8 + 4 * m * n, d2h_calls=2))


def validate_verification(value, case):
    indices = point_indices(case)
    require(value["all_finite"] is True and digest(value["output_sha256"])
            and identical(value["checked_elements"], len(indices) if indices else case["output_elements"]),
            "finite 检查、输出摘要或数值抽样数量不符")
    require(finite(value["max_absolute"]) and value["max_absolute"] >= 0
            and finite(value["max_tolerance_ratio"]) and 0 <= value["max_tolerance_ratio"] <= 1,
            "数值误差或固定容差无效")
    require(isinstance(value["points"], list)
            and identical([p["index"] for p in value["points"]], indices), "FP64 抽样位置不符")
    if indices:
        require(value["reference_sha256"] is None, "抽样数值模式不能冒充全量对照")
        errors, ratios = [], []
        for point in value["points"]:
            require(finite(point["actual"]) and finite(point["reference"]), "数值抽样含非有限值")
            error = abs(point["actual"] - point["reference"])
            limit = 2e-4 + 2e-4 * abs(point["reference"])
            require(error <= limit, "FP64 抽样不满足冻结容差")
            errors.append(error)
            ratios.append(error / limit)
        require(abs(max(errors) - value["max_absolute"]) <= 1e-12
                and abs(max(ratios) - value["max_tolerance_ratio"]) <= 1e-12, "抽样数值汇总与原始值不符")
    else:
        require(digest(value["reference_sha256"]), "全量 FP64 参照摘要缺失")


def validate_report(report, spec):
    require(identical(report["schema_version"], 1) and report["benchmark"] == BENCHMARK
            and report["status"] == "passed" and report["input_sha256"] == INPUT_SHA256
            and report["model_sha256"] == MODEL_SHA256, "micro 报告状态或身份无效")
    require(integer(report["trial"]) and report["trial"] < 5
            and identical(report["protocol"], spec["measurement"]), "trial 或计时协议无效")
    require(identical(report["dimensions"], DIMENSIONS) and identical(report["arithmetic"], ARITHMETIC),
            "模型维度或算术配置不符")
    d = report["device"]
    require(isinstance(d["name"], str) and d["name"] and isinstance(d["uuid"], str) and len(d["uuid"]) == 36
            and isinstance(d["compute_capability"], list) and len(d["compute_capability"]) == 2
            and all(integer(v) for v in d["compute_capability"])
            and all(integer(d[k], 1) for k in ("driver_version", "runtime_version", "cublas_version")), "设备身份无效")
    plan, payload, _ = common.memory_plan(report)
    require(identical(plan, report["memory_plan"]) and identical(report["owned_device_bytes"], plan["total_owned_bytes"])
            and identical(report["weight_h2d_bytes"], payload)
            and identical(report["rope_h2d_bytes"], plan["rope_bytes"]), "内存计划或初始化传输不符")
    for key in ("model_load_ns", "storage_initialization_ns", "weight_decode_upload_ns", "kv_initialization_ns"):
        require(integer(report[key], 1), "初始化时间无效")
    require(report["weight_decode_upload_ns"] <= report["storage_initialization_ns"], "权重上传不属于存储初始化区间")
    steady = dict(ZERO_ALLOCATIONS, allocation_calls=4, allocations=4, allocated_bytes=plan["total_owned_bytes"])
    common.allocation_snapshot(report["before_initialization_allocations"], ZERO_ALLOCATIONS)
    for key in ("before_cases_allocations", "after_cases_allocations"):
        common.allocation_snapshot(report[key], steady)
    common.allocation_snapshot(report["after_destruction_allocations"], dict(steady, release_calls=4, releases=4))
    require(identical(report["kv_initialization_transfers"],
                      transfers(h2d_bytes=4 * 2048 * (2 * 1024 * 4 + 2 * 4), d2h_bytes=8, h2d_calls=256, d2h_calls=1)),
            "layer 0 全容量 KV 初始化传输不符")
    planned = make_cases(spec, report["dimensions"])
    if report["trial"] % 2:
        planned.reverse()
    require(len(report["cases"]) == len(planned), "micro 用例不完整")
    weights = {w["name"]: w for w in report["weights"]}
    result = {}
    for actual, case in zip(report["cases"], planned):
        require(all(identical(actual[key], value) for key, value in case.items()), "实际形状、输入或用例顺序不符")
        require(actual["status"] == "passed" and actual["input_sha256"] == case_input_hash(case), "用例失败或输入 recipe 摘要不符")
        if case["tensor"]:
            shape = [case["n"], case["k"]] if case["operation"] == "matrix" else [1, case["group_width"]]
            require(identical(weights[case["tensor"]]["shape"], shape), "用例与实际权重形状不符")
        common.allocation_snapshot(actual["before_allocations"], steady)
        common.allocation_snapshot(actual["after_allocations"], steady)
        require(integer(actual["preparation_host_ns"], 1), "用例准备时间无效")
        copy = expected_transfers(case)
        require(identical(actual["preparation_transfers"], copy["preparation"]), "准备阶段传输不符")
        require(len(actual["samples"]) == 5, "warmup 或 measured 样本缺失")
        host, device, verifications = [], [], []
        for index, sample in enumerate(actual["samples"]):
            require(identical(sample["iteration"], index) and sample["phase"] == ("warmup" if index < 2 else "measured")
                    and identical(sample["calls"], 32), "样本顺序、阶段或 API 调用数量不符")
            for key in ("setup_host_ns", "host_enqueue_to_completion_ns", "validation_host_ns"):
                require(integer(sample[key], 1), "host 计时字段无效")
            require(finite(sample["device_interval_ms"]) and sample["device_interval_ms"] > 0, "CUDA event 区间无效")
            for phase in ("setup", "measured", "validation"):
                require(identical(sample[phase + "_transfers"], copy[phase]), "计时边界或显式传输计数不符")
            validate_verification(sample["verification"], case)
            verifications.append(sample["verification"])
            if index >= 2:
                host.append(sample["host_enqueue_to_completion_ns"] / 32)
                device.append(sample["device_interval_ms"] * 1e6 / 32)
        require(all(identical(v, verifications[0]) for v in verifications), "相同输入的重复输出或参照发生变化")
        result[case["name"]] = dict(host_ns_per_call=host, device_ns_per_call=device,
                                   verification=verifications[0])
    return dict(trial=report["trial"], cases=result, samples=1875, api_calls=60000,
                measured_samples=1125, measured_api_calls=36000, owned_device_bytes=plan["total_owned_bytes"])


BOOTSTRAP_RANKS = sorted(sorted(indices)[2] for indices in itertools.product(range(5), repeat=5))


def trial_statistics(samples):
    require(len(samples) == 5 and all(len(v) == 3 and all(finite(x) and x > 0 for x in v) for v in samples),
            "统计必须使用 5 个独立 trial，每个 trial 含 3 个实测样本")
    medians = [statistics.median(v) for v in samples]
    ordered = sorted(medians)
    resamples = [ordered[index] for index in BOOTSTRAP_RANKS]
    center = statistics.median(medians)
    return dict(samples_ns_per_call=samples, trial_median_ns_per_call=medians, median_ns_per_call=center,
                median_ci95_ns_per_call=[common.percentile(resamples, q) for q in (0.025, 0.975)],
                mad_percent=100 * statistics.median(abs(v - center) for v in medians) / center,
                range_ns_per_call=[min(medians), max(medians)])


def summarize(reports, spec):
    require(identical([r["trial"] for r in reports], list(range(5))), "独立 trial 缺失、重复或顺序不符")
    audited = [validate_report(report, spec) for report in reports]
    for key in ("device", "weights", "memory_plan", "dimensions", "arithmetic"):
        require(all(identical(report[key], reports[0][key]) for report in reports), f"进程间身份不一致：{key}")
    cases = []
    for case in make_cases(spec, reports[0]["dimensions"]):
        values = [report["cases"][case["name"]] for report in audited]
        require(all(identical(v["verification"], values[0]["verification"]) for v in values), "trial 间数值输出发生变化")
        host = trial_statistics([v["host_ns_per_call"] for v in values])
        device = trial_statistics([v["device_ns_per_call"] for v in values])
        flops = case["logical_flops_per_call"]
        cases.append(dict(case, host=host, device_interval=device, verification=values[0]["verification"],
                          logical_gflops_on_device_interval=flops / device["median_ns_per_call"] if flops else None))
    return dict(schema_version=1, benchmark=BENCHMARK, status="measured", independent_trials=5,
                case_count=375, raw_samples=9375, measured_samples=5625, api_calls=300000, measured_api_calls=180000,
                statistics=STATISTICS, data_path_gates="passed", cases=cases,
                end_to_end_speedup=None, hardware_dram_bandwidth=None, profiler_complete=False, gpu_serving=False)


def validate_bundle(directory):
    manifest_path = artifact(directory, "manifest.json")
    manifest = read(manifest_path)
    require(identical(manifest["schema_version"], 1) and manifest["benchmark"] == BENCHMARK
            and manifest["protocol_id"] == PROTOCOL_ID and identical(manifest["reports"], schedule())
            and identical(manifest["statistics"], STATISTICS), "manifest 或完整独立 trial 计划不符")
    require(manifest["dependencies"]["llama_commit"] == common.LLAMA_COMMIT
            and manifest["build"]["own_cuda"] == "ON" and manifest["build"]["upstream_cuda"] == "OFF"
            and manifest["build"]["type"] in ("Release", "RelWithDebInfo"), "构建配置或依赖不符")
    require(manifest["model"]["sha256"] == MODEL_SHA256 and digest(manifest["binary"]["sha256"])
            and manifest["input"]["sha256"] == INPUT_SHA256, "模型、二进制或输入身份不符")
    files = common.source_archive(directory, manifest["source"])
    require(all(name in files for name in ("apps/cuda_kernel_bench.cpp", "apps/cuda_micro_protocol.h",
                "src/minillm/cuda/attention.cu", "scripts/analyze_cuda_micro.py")), "缺少 micro 源码")
    require(files["benchmarks/runtime-inputs/qwen3-cuda-micro-v0.json"]["sha256"] == INPUT_SHA256,
            "源码中的 micro 输入不符")
    availability, seen = [], set()

    def check(name, expected=None):
        require(name not in seen, "重复 artifact")
        path = artifact(directory, name)
        actual = sha(path)
        require(expected is None or digest(expected) and actual == expected, f"artifact 摘要不符：{name}")
        seen.add(name)
        availability.append(dict(locator=name, mandatory=True, exists=True, sha256=actual,
                                 size_bytes=path.stat().st_size, status="AVAILABLE"))
        return path

    for name in ("manifest.json", "collection-status.json", manifest["source"]["state_file"], manifest["source"]["snapshot"]["path"]):
        check(name)
    spec_path = check(manifest["input"]["path"], INPUT_SHA256)
    spec = read(spec_path)
    for item in manifest["artifacts"]:
        check(item["path"], item["sha256"])
    for local, source in (("verify.py", "scripts/analyze_cuda_micro.py"), ("analyze_cuda_benchmark.py", "scripts/analyze_cuda_benchmark.py")):
        require(local in seen and sha(artifact(directory, local)) == files[source]["sha256"], "归档复核工具不属于采集源码")
    collection = read(artifact(directory, "collection-status.json"))
    require(collection["status"] == "passed" and identical(collection["planned_reports"], 5)
            and len(collection["reports"]) == 5, "采集失败或缺少独立进程")
    reports = []
    for slot, record in zip(schedule(), collection["reports"]):
        require(record["file"] == slot["file"] and identical(record["exit_code"], 0), "进程顺序或状态不符")
        report = read(check(record["file"], record["sha256"]))
        require(identical(report["trial"], slot["trial"]) and identical(report["run_identity"], dict(
            run_id=manifest["run_id"], manifest_sha256=sha(manifest_path), binary_sha256=manifest["binary"]["sha256"],
            source_state_sha256=manifest["source"]["worktree_state_sha256"])), "进程未绑定本次 manifest、源码或二进制")
        for item in record["artifacts"]:
            check(item["path"], item["sha256"])
        require({slot["file"] + suffix for suffix in (".stdout.txt", ".stderr.txt", ".process.json")} <= seen,
                "缺少原始 stdout、stderr 或进程环境")
        process = read(artifact(directory, slot["file"] + ".process.json"))
        require(identical(process["trial"], slot["trial"]) and identical(process["exit_code"], 0)
                and isinstance(process["before"], dict) and isinstance(process["after"], dict)
                and isinstance(process["arguments"], list) and process["arguments"], "进程环境或原始命令无效")
        reports.append(report)
    summary = summarize(reports, spec)
    summary.update(run_id=manifest["run_id"], source_state_sha256=manifest["source"]["worktree_state_sha256"])
    return dict(summary=summary, availability=dict(schema_version=1, status="AVAILABLE", artifacts=availability),
                weights=reports[0]["weights"], memory=reports[0]["memory_plan"])


def analysis_text(summary):
    lines = ["# 自有 CUDA 真实形状微基准", "",
             "- 375 个用例，5 个独立进程；每项 2 次 warmup、3 次测量，每个样本连续调用 32 次 API。",
             "- 表中数值为每次 API 调用的区间均摊值，不是单个 kernel 的纯执行时间；CUDA events 包含提交空隙。",
             "- 初始化、输入重置、状态/输出下载和 FP64 对照均位于计时外；RoPE 每个样本从原输入开始连续旋转 32 次。",
             "- 矩阵和 attention 保留 FP64 抽样值，其余算子逐元素验证；所有输出检查 finite，原始报告保留全部样本。",
             "- 区间使用 5^5 次独立 trial 中位数重采样。仅 5 个 trial，区间稳定性有限；没有挑选最快进程。",
             "- 显式传输和设备分配仅统计项目路径，不包含 NVIDIA 库内部资源；未测量硬件 DRAM 带宽。",
             "- 本报告没有 CPU 对照、模型加速、GPU Serving 或自有 PagedAttention 结论。", "",
             "| 用例 | M | N | K | host us/call | device interval us/call | device 95% 区间 us | device MAD % |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
    for case in summary["cases"]:
        host, device = case["host"], case["device_interval"]
        lo, hi = device["median_ci95_ns_per_call"]
        lines.append(f"| {case['name']} | {case['m']} | {case['n']} | {case['k']} | "
                     f"{host['median_ns_per_call']/1000:.3f} | {device['median_ns_per_call']/1000:.3f} | "
                     f"[{lo/1000:.3f}, {hi/1000:.3f}] | {device['mad_percent']:.2f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--directory")
    mode.add_argument("--report")
    mode.add_argument("--schedule", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    try:
        if args.schedule:
            result = dict(reports=schedule(), statistics=STATISTICS)
        elif args.report:
            require(args.input and sha(args.input) == INPUT_SHA256, "单进程复核需要冻结的 --input")
            audit = validate_report(read(args.report), read(args.input))
            result = {key: value for key, value in audit.items() if key != "cases"}
            result.update(status="passed", case_count=len(audit["cases"]))
        else:
            result = validate_bundle(args.directory)
            if args.write:
                publish(args.directory, {"availability.json": result["availability"], "weight-plan.json": result["weights"],
                    "memory-plan.json": result["memory"], "analysis.md": analysis_text(result["summary"]),
                    "summary.json": result["summary"]})
            result = dict(status=result["summary"]["status"], case_count=result["summary"]["case_count"],
                          independent_trials=5, measured_samples=result["summary"]["measured_samples"],
                          artifacts=len(result["availability"]["artifacts"]))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValidationError, OSError, ValueError, KeyError, TypeError, IndexError, zipfile.BadZipFile) as error:
        print(f"CUDA 微基准证据复核失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
