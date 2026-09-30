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
PRECISION_INPUT_SHA256 = "3b9ec80ee5e9cc83865378f21c46d5dedf4975530e7686a1dfb61d5f4af992b8"
PRECISION_PROTOCOL = "precision-experiment-v1"
PRECISION_MODES = ("f32-pedantic", "f16-matrix-f32acc")
PRECISION_STATISTICS = dict(independent_trials=3, process_median_repeats=3, calls_per_sample=20,
    statistics_unit="independent_trial_median", comparison=True, confidence_interval=None,
    hardware_dram_bandwidth=None)
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
KV_STATISTICS = dict(independent_trials=3, process_median_repeats=3, calls_per_sample=32,
    statistics_unit="independent_trial_median", relative_difference="100*(paged/contiguous-1)",
    comparison=True, confidence_interval=None, hardware_dram_bandwidth=None)


def schedule(precision=False, gpu_kv=False):
    require(not (precision and gpu_kv), "精度与分页实验不能同时选择")
    if gpu_kv:
        return [dict(trial=trial, kv_layout=layout, file=f"micro-t{trial}-{layout}.json",
                     order="reverse" if trial % 2 else "canonical")
                for trial in range(3) for layout in (common.KV_LAYOUTS[::-1] if trial % 2 else common.KV_LAYOUTS)]
    if precision:
        return [dict(trial=trial, precision_mode=mode, file=f"micro-t{trial}-{mode}.json",
                     order="reverse" if trial % 2 else "canonical")
                for trial in range(3) for mode in (PRECISION_MODES[::-1] if trial % 2 else PRECISION_MODES)]
    return [dict(trial=i, file=f"cuda-micro-t{i}.json", order="reverse" if i % 2 else "canonical") for i in range(5)]


def sample_rows(count):
    return sorted({0, count // 2, count - 1})


def sample_columns(count):
    return list(range(count)) if count <= 8 else [i * (count - 1) // 7 for i in range(8)]


def make_cases(spec, d):
    if spec["protocol_id"] == common.KV_PROTOCOL:
        require(identical(spec["micro"]["rows"], [1, 4, 32]) and
                identical(spec["micro"]["effective_contexts"], [128, 1536]), "分页 micro 必须为六类冻结形状")
        result = []
        for m in (1, 4, 32):
            for length in (128, 1536):
                role = "prefill" if m == 32 else "decode"
                width = d["heads"] * d["head_dim"]
                result.append(dict(name=f"attention-{role}-m{m}-l{length}", operation="attention", role=role,
                    tensor="", m=m, n=width, k=0, group_width=d["head_dim"], max_context=length,
                    slots=list(range(4)) if m == 4 else [0] * m,
                    positions=list(range(length - m, length)) if m == 32 else [length - 1] * m,
                    input_seed=spec["micro"]["seed"] + len(result) + 1, logical_flops_per_call=None,
                    output_elements=m * width, query_heads=d["heads"], kv_heads=d["kv_heads"],
                    head_dim=d["head_dim"], kv_max_length=2048))
        return result
    if spec["protocol_id"] == PRECISION_PROTOCOL:
        # 借用既有形状生成器，仅保留冻结的矩阵子集和原 seed。
        legacy = dict(matrix=dict(rows=spec["micro"]["rows"], roles=[m["role"] for m in spec["micro"]["matrices"]]),
                      ops=dict(rms_norm_roles=[], rope_roles=[], attention_contexts=[], mixed=dict(prefix_tokens=[])))
        result = _make_cases(legacy, d)
        for i, matrix in enumerate(spec["micro"]["matrices"]):
            for j, m in enumerate(spec["micro"]["rows"]):
                case = result[i * 4 + j]
                require((case["m"], case["n"], case["k"]) == (m, matrix["N"], matrix["K"]), "precision shape 不符")
                case["input_seed"] = matrix["seeds"][j]
        require(len(result) == 16, "precision 必须包含 16 个 shape")
        return result
    result = _make_cases(spec, d)
    require(len(result) == 375 and len({v["name"] for v in result}) == 375, "micro 用例集合无效")
    return result


def _make_cases(spec, d):
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

def precision_arithmetic(mode, spec):
    half = mode == PRECISION_MODES[1]
    return dict(ARITHMETIC, device_weight_dtype="F16_matrices_F32_norms" if half else "F32",
                matrix_operand_dtype="F16" if half else "F32", matrix_accumulation_dtype="F32",
                matrix_output_dtype="F32", gemm_compute="CUBLAS_COMPUTE_32F" if half else ARITHMETIC["gemm_compute"],
                math_mode=spec["precision"]["candidate" if half else "baseline"]["math_mode"],
                tensor_core_usage="unverified")


def validate_precision_verification(value, case):
    require(value["passed"] is True and value["all_finite"] is True and value["first_nonfinite"] is None
            and identical(value["checked_elements"], case["output_elements"]), "precision 未完成全量 finite/FP64 验证")
    for key in ("reference_sha256", "absolute_sums_sha256", "output_sha256"):
        require(digest(value[key]), "precision 数值摘要缺失")
    for key in ("max_absolute", "max_tolerance_ratio", "rmse"):
        require(finite(value[key]) and value[key] >= 0, "precision 数值误差无效")
    require(value["max_tolerance_ratio"] <= 1 and value["rmse"] <= value["max_absolute"] + 1e-12,
            "precision 数值门槛失败")
    gamma = case["k"] * 2**-24 / (1 - case["k"] * 2**-24)
    for name in ("worst_absolute", "worst_ratio"):
        p = value[name]
        require(integer(p["index"]) and p["index"] < case["output_elements"]
                and all(finite(p[k]) for k in ("actual", "reference", "sum_absolute_products"))
                and p["sum_absolute_products"] >= abs(p["reference"]), "precision 最坏元素无效")
        error = abs(p["actual"] - p["reference"])
        limit = 2e-4 + 2e-4 * abs(p["reference"]) + 4 * gamma * p["sum_absolute_products"]
        require(error <= limit and error <= value["max_absolute"] + 1e-12, "precision FP64 元素超出固定界限")
        actual, reported = (error, value["max_absolute"]) if name == "worst_absolute" else (
            error / limit, value["max_tolerance_ratio"])
        require(abs(actual - reported) <= 1e-12, "precision 最坏元素与摘要不符")


def validate_report(report, spec):
    precision = spec["protocol_id"] == PRECISION_PROTOCOL
    gpu_kv = spec["protocol_id"] == common.KV_PROTOCOL
    mode = report.get("precision_mode") if precision else None
    require(not precision or mode in PRECISION_MODES and report["protocol_id"] == PRECISION_PROTOCOL, "精度模式或协议不符")
    layout = report.get("kv_layout") if gpu_kv else "contiguous"
    if gpu_kv:
        require(layout in common.KV_LAYOUTS and report["protocol_id"] == common.KV_PROTOCOL
                and report["precision_mode"] == "f32-pedantic"
                and report["scope"] in ("paired_kv_micro", "diagnostic_single_process"), "分页协议、精度或布局不符")
    calls = 20 if precision else 32
    input_hash = common.KV_INPUT_SHA256 if gpu_kv else PRECISION_INPUT_SHA256 if precision else INPUT_SHA256
    require(identical(report["schema_version"], 2 if precision else 1) and report["benchmark"] == BENCHMARK
            and report["status"] == "passed" and report["input_sha256"] == input_hash
            and report["model_sha256"] == MODEL_SHA256, "micro 报告状态或身份无效")
    require(integer(report["trial"]) and report["trial"] < (3 if precision or gpu_kv else 5)
            and identical(report["protocol"], spec["measurement"]), "trial 或计时协议无效")
    require(identical(report["dimensions"], DIMENSIONS) and
            identical(report["arithmetic"], precision_arithmetic(mode, spec) if precision else ARITHMETIC),
            "模型维度或算术配置不符")
    d = report["device"]
    require(isinstance(d["name"], str) and d["name"] and isinstance(d["uuid"], str) and len(d["uuid"]) == 36
            and isinstance(d["compute_capability"], list) and len(d["compute_capability"]) == 2
            and all(integer(v) for v in d["compute_capability"])
            and all(integer(d[k], 1) for k in ("driver_version", "runtime_version", "cublas_version")), "设备身份无效")
    plan, payload, _ = common.memory_plan(report, precision_mode=mode, kv_layout=layout)
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
    require(identical(report["kv_initialization_transfers"], transfers() if precision else
                      transfers(h2d_bytes=4 * 2048 * (2 * 1024 * 4 + 2 * 4), d2h_bytes=8, h2d_calls=256, d2h_calls=1)),
            "layer 0 全容量 KV 初始化传输不符")
    if precision:
        verification = report["weight_validation"]
        require(verification["status"] == "passed" and identical(verification["unique_tensors"], 310)
                and identical(verification["d2h_bytes"], payload)
                and integer(verification["staging_peak_bytes"], 1) and verification["staging_peak_bytes"] <= 8*1024**2,
                "全量设备载荷验证或 staging 超出上限")
    table_probe = validate_table_probe(report) if gpu_kv else None
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
        require(len(actual["samples"]) == (10 if precision else 5), "warmup 或 measured 样本缺失")
        host, device, verifications = [], [], []
        boundaries = {b: dict(host_ns_per_call=[], device_ns_per_call=[]) for b in ("gemm_only", "cast_inclusive")}
        for index, sample in enumerate(actual["samples"]):
            iteration = index // 2 if precision else index
            require(identical(sample["iteration"], iteration) and sample["phase"] == ("warmup" if iteration < 2 else "measured")
                    and identical(sample["calls"], calls), "样本顺序、阶段或 API 调用数量不符")
            if precision:
                boundary = "gemm_only" if index % 2 == 0 else "cast_inclusive"
                half = mode == PRECISION_MODES[1]
                require(sample["boundary"] == boundary and identical(sample["setup_cast_calls"], int(half and index % 2 == 0))
                        and identical(sample["measured_cast_calls"], calls if half and index % 2 else 0)
                        and identical(sample["device_status"], [0, 2**31-1]), "转换计时边界或 device status 不符")
            for key in ("setup_host_ns", "host_enqueue_to_completion_ns", "validation_host_ns"):
                require(integer(sample[key], 1), "host 计时字段无效")
            require(finite(sample["device_interval_ms"]) and sample["device_interval_ms"] > 0, "CUDA event 区间无效")
            for phase in ("setup", "measured", "validation"):
                require(identical(sample[phase + "_transfers"], copy[phase]), "计时边界或显式传输计数不符")
            (validate_precision_verification if precision else validate_verification)(sample["verification"], case)
            verifications.append(sample["verification"])
            if iteration >= 2:
                h, t = sample["host_enqueue_to_completion_ns"] / calls, sample["device_interval_ms"] * 1e6 / calls
                host.append(h); device.append(t)
                if precision:
                    boundaries[boundary]["host_ns_per_call"].append(h)
                    boundaries[boundary]["device_ns_per_call"].append(t)
        require(all(identical(v, verifications[0]) for v in verifications), "相同输入的重复输出或参照发生变化")
        result[case["name"]] = dict(host_ns_per_call=host, device_ns_per_call=device,
                                   verification=verifications[0])
        if precision:
            result[case["name"]]["boundaries"] = boundaries
    samples = len(planned) * (10 if precision else 5)
    audited = dict(trial=report["trial"], cases=result, samples=samples, api_calls=samples*calls,
                   measured_samples=samples*3//5, measured_api_calls=samples*3//5*calls,
                   owned_device_bytes=plan["total_owned_bytes"])
    if gpu_kv:
        audited["table_probe"] = table_probe
    return audited


def validate_table_probe(report):
    probe = report["table_upload_probe"]
    if report["kv_layout"] == "contiguous":
        require(probe is None and identical(report["page_table_h2d_bytes"], 0)
                and report.get("page_table_sha256") is None, "连续布局不能伪造页表传输")
        return None
    expected = hashlib.sha256(struct.pack("<512i", *reversed(range(512)))).hexdigest()
    require(report["page_table_sha256"] == expected and identical(probe["bytes_per_call"], 2048)
            and identical(report["page_table_h2d_bytes"], 2048 * (1 + 5 * 32))
            and len(probe["samples"]) == 5, "分页映射、完整上传或 probe 样本不符")
    result = dict(host_ns_per_call=[], device_ns_per_call=[])
    for i, sample in enumerate(probe["samples"]):
        require(identical(sample["iteration"], i) and sample["phase"] == ("warmup" if i < 2 else "measured")
                and identical(sample["calls"], 32) and identical(sample["h2d_bytes"], 2048*32)
                and integer(sample["host_enqueue_to_completion_ns"], 1)
                and finite(sample["device_interval_ms"]) and sample["device_interval_ms"] > 0,
                "表上传 probe 的边界、字节数或时间不符")
        if i >= 2:
            result["host_ns_per_call"].append(sample["host_enqueue_to_completion_ns"]/32)
            result["device_ns_per_call"].append(sample["device_interval_ms"]*1e6/32)
    return result


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
    if spec["protocol_id"] == common.KV_PROTOCOL:
        return summarize_kv(reports, spec)
    if spec["protocol_id"] == PRECISION_PROTOCOL:
        return summarize_precision(reports, spec)
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

def summarize_precision(reports, spec):
    planned = schedule(True)
    require(identical([(r["trial"], r["precision_mode"]) for r in reports],
                      [(s["trial"], s["precision_mode"]) for s in planned]), "precision 配对 trial 缺失或顺序不符")
    audited = [validate_report(report, spec) for report in reports]
    for key in ("device", "dimensions"):
        require(all(identical(r[key], reports[0][key]) for r in reports), f"precision 进程身份不一致：{key}")
    by_mode = {mode: [i for i, r in enumerate(reports) if r["precision_mode"] == mode] for mode in PRECISION_MODES}
    for indices in by_mode.values():
        for key in ("weights", "memory_plan", "arithmetic"):
            require(all(identical(reports[i][key], reports[indices[0]][key]) for i in indices), "同精度身份发生变化")
    base, candidate = reports[0], reports[1]
    source_keys = ("name", "shape", "source_dtype", "effective_sha256", "alias_of")
    require(identical([{k: w[k] for k in source_keys} for w in base["weights"]],
                      [{k: w[k] for k in source_keys} for w in candidate["weights"]]), "跨精度源有效权重不一致")
    cases = []
    for case in make_cases(spec, DIMENSIONS):
        modes = {mode: [audited[i]["cases"][case["name"]] for i in indices] for mode, indices in by_mode.items()}
        for values in modes.values():
            require(all(identical(v["verification"], values[0]["verification"]) for v in values),
                    "同精度跨 trial 输出不一致")
        comparisons = {}
        for boundary in ("gemm_only", "cast_inclusive"):
            comparisons[boundary] = {}
            for clock in ("host_ns_per_call", "device_ns_per_call"):
                samples = {mode: [v["boundaries"][boundary][clock] for v in values] for mode, values in modes.items()}
                medians = {mode: [statistics.median(v) for v in values] for mode, values in samples.items()}
                gains = [1-c/b for b, c in zip(medians[PRECISION_MODES[0]], medians[PRECISION_MODES[1]])]
                comparisons[boundary][clock] = dict(samples=samples, trial_medians=medians, paired_gains=gains,
                                                     median_paired_gain=statistics.median(gains))
        cases.append(dict(case, comparisons=comparisons,
                          verification={mode: values[0]["verification"] for mode, values in modes.items()}))
    owned = {mode: reports[indices[0]]["owned_device_bytes"] for mode, indices in by_mode.items()}
    reduction = 1-owned[PRECISION_MODES[1]]/owned[PRECISION_MODES[0]]
    return dict(schema_version=2, benchmark=BENCHMARK, protocol_id=PRECISION_PROTOCOL, status="micro_measured",
                independent_trials=3, process_count=6, case_count=16, raw_samples=960, measured_samples=576,
                api_calls=19200, measured_api_calls=11520, statistics=PRECISION_STATISTICS,
                data_path_gates="passed", cases=cases, owned_device_bytes=owned,
                memory_reduction=reduction, memory_gate=reduction >= spec["gates"]["owned_device_bytes_reduction_min"],
                end_to_end_speedup=None, product_eligible=False, tensor_core_usage="unverified",
                full_model_numerics=False, gpu_serving=False)


def summarize_kv(reports, spec):
    planned = schedule(gpu_kv=True)
    require(identical([(r["trial"], r["kv_layout"]) for r in reports],
                      [(s["trial"], s["kv_layout"]) for s in planned])
            and all(r["scope"] == "paired_kv_micro" for r in reports), "分页配对进程不完整或使用了诊断运行")
    audited = [validate_report(report, spec) for report in reports]
    for key in ("device", "dimensions", "weights", "arithmetic"):
        require(all(identical(r[key], reports[0][key]) for r in reports), f"分页两臂身份不一致：{key}")
    indices = {layout: [i for i, r in enumerate(reports) if r["kv_layout"] == layout] for layout in common.KV_LAYOUTS}
    cases = []
    for case in make_cases(spec, DIMENSIONS):
        values = [r["cases"][case["name"]] for r in audited]
        require(all(identical(v["verification"], values[0]["verification"]) for v in values),
                "跨布局或 trial 的输出不一致")
        comparisons = {clock: common.kv_paired_statistics({
            layout: [values[i][clock] for i in rows] for layout, rows in indices.items()})
            for clock in ("host_ns_per_call", "device_ns_per_call")}
        cases.append(dict(case, comparisons=comparisons, verification=values[0]["verification"]))
    table = {}
    for clock in ("host_ns_per_call", "device_ns_per_call"):
        samples = [audited[i]["table_probe"][clock] for i in indices["paged"]]
        medians = [statistics.median(row) for row in samples]
        table[clock] = dict(samples=samples, trial_medians=medians, median=statistics.median(medians))
    return dict(schema_version=1, benchmark=BENCHMARK, protocol_id=common.KV_PROTOCOL, status="micro_measured",
        independent_trials=3, process_count=6, case_count=6, raw_samples=180, measured_samples=108,
        api_calls=5760, measured_api_calls=3456, statistics=KV_STATISTICS, data_path_gates="passed",
        cases=cases, table_upload_probe=dict(bytes_per_call=2048, measurements=table),
        owned_device_bytes={layout: audited[rows[0]]["owned_device_bytes"] for layout, rows in indices.items()},
        end_to_end_speedup=None, product_eligible=False, gpu_serving=False)


def validate_bundle(directory):
    manifest_path = artifact(directory, "manifest.json")
    manifest = read(manifest_path)
    precision = manifest["protocol_id"] == PRECISION_PROTOCOL
    gpu_kv = manifest["protocol_id"] == common.KV_PROTOCOL
    input_hash = common.KV_INPUT_SHA256 if gpu_kv else PRECISION_INPUT_SHA256 if precision else INPUT_SHA256
    plan = schedule(precision, gpu_kv)
    require(identical(manifest["schema_version"], 2 if precision else 1) and manifest["benchmark"] == BENCHMARK
            and manifest["protocol_id"] == (common.KV_PROTOCOL if gpu_kv else PRECISION_PROTOCOL if precision else PROTOCOL_ID)
            and identical(manifest["reports"], plan)
            and identical(manifest["statistics"], KV_STATISTICS if gpu_kv else PRECISION_STATISTICS if precision else STATISTICS),
            "manifest 或完整独立 trial 计划不符")
    require(manifest["dependencies"]["llama_commit"] == common.LLAMA_COMMIT
            and manifest["build"]["own_cuda"] == "ON" and manifest["build"]["upstream_cuda"] == "OFF"
            and manifest["build"]["type"] in ("Release", "RelWithDebInfo"), "构建配置或依赖不符")
    require(manifest["model"]["sha256"] == MODEL_SHA256 and digest(manifest["binary"]["sha256"])
            and manifest["input"]["sha256"] == input_hash, "模型、二进制或输入身份不符")
    files = common.source_archive(directory, manifest["source"])
    require(all(name in files for name in ("apps/cuda_kernel_bench.cpp", "apps/cuda_micro_protocol.h",
                "src/minillm/cuda/attention.cu", "scripts/analyze_cuda_micro.py")), "缺少 micro 源码")
    input_name = "qwen3-gpu-kv-v1.json" if gpu_kv else "qwen3-precision-v1.json" if precision else "qwen3-cuda-micro-v0.json"
    require(files[f"benchmarks/runtime-inputs/{input_name}"]["sha256"] == input_hash,
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
    spec_path = check(manifest["input"]["path"], input_hash)
    spec = read(spec_path)
    for item in manifest["artifacts"]:
        check(item["path"], item["sha256"])
    for local, source in (("verify.py", "scripts/analyze_cuda_micro.py"), ("analyze_cuda_benchmark.py", "scripts/analyze_cuda_benchmark.py")):
        require(local in seen and sha(artifact(directory, local)) == files[source]["sha256"], "归档复核工具不属于采集源码")
    collection = read(artifact(directory, "collection-status.json"))
    require(collection["status"] == "passed" and identical(collection["planned_reports"], len(plan))
            and len(collection["reports"]) == len(plan), "采集失败或缺少独立进程")
    reports = []
    for slot, record in zip(plan, collection["reports"]):
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
        if precision:
            require(report["precision_mode"] == slot["precision_mode"]
                    and process["precision_mode"] == slot["precision_mode"], "进程 precision 身份不符")
            arguments = process["arguments"]
            require(arguments.count("--cuda-precision") == 1
                    and arguments[arguments.index("--cuda-precision")+1] == slot["precision_mode"],
                    "原始命令未选择声明的 precision")
        if gpu_kv:
            arguments = process["arguments"]
            require(report["scope"] == "paired_kv_micro" and report["kv_layout"] == slot["kv_layout"]
                    and process["kv_layout"] == slot["kv_layout"] and process["precision_mode"] == "f32-pedantic",
                    "分页原始进程的布局或精度不符")
            require(arguments.count("--kv-layout") == 1 and
                    arguments[arguments.index("--kv-layout")+1] == slot["kv_layout"],
                    "原始命令没有选择声明的 KV 布局")
        reports.append(report)
    summary = summarize(reports, spec)
    summary.update(run_id=manifest["run_id"], source_state_sha256=manifest["source"]["worktree_state_sha256"])
    return dict(summary=summary, availability=dict(schema_version=1, status="AVAILABLE", artifacts=availability),
                weights={r["precision_mode"]: r["weights"] for r in reports} if precision else reports[0]["weights"],
                memory={r["kv_layout"]: r["memory_plan"] for r in reports} if gpu_kv else
                       {r["precision_mode"]: r["memory_plan"] for r in reports} if precision else reports[0]["memory_plan"])


def analysis_text(summary):
    if summary.get("protocol_id") == common.KV_PROTOCOL:
        lines = ["# GPU KV 同容量微基准", "",
            "- 同一 GPU、F32 权重和二进制；两布局各三个独立 trial，每项两次预热、三次测量。",
            "- 每样本 32 次 attention API；统计单位为独立进程的样本中位数。",
            "- 差异为 100*(paged/contiguous-1)，正数表示退化；不宣称统计显著性。",
            "- 两臂 KV 均为 896 MiB；分页额外拥有 2 KiB 设备表，不构成显存节省。",
            "- 表上传单独计量；完整分配、映射维护和上传成本须由模型/Serving 计时验证。",
            "- CUDA event 区间含提交空隙，不是纯 kernel 时间；没有端到端加速或采用结论。", "",
            "| 用例 | 连续 host us | 分页 host us | host 差异 % | 各轮 host 差异 % | event 差异 % |",
            "| --- | ---: | ---: | ---: | --- | ---: |"]
        for case in summary["cases"]:
            host, device = (case["comparisons"][key] for key in ("host_ns_per_call", "device_ns_per_call"))
            values = host["trial_medians"]
            changes = ", ".join(f"{v:.2f}" for v in host["paired_relative_percent"])
            lines.append(f"| {case['name']} | {statistics.median(values['contiguous'])/1000:.3f} | "
                         f"{statistics.median(values['paged'])/1000:.3f} | {host['median_relative_percent']:.2f} | "
                         f"{changes} | {device['median_relative_percent']:.2f} |")
        probe = summary["table_upload_probe"]["measurements"]
        lines += ["", f"2 KiB table probe 的 host/event 每次均摊中位数分别为 "
                  f"{probe['host_ns_per_call']['median']/1000:.3f}/"
                  f"{probe['device_ns_per_call']['median']/1000:.3f} us。"]
        return "\n".join(lines) + "\n"
    if summary.get("protocol_id") == PRECISION_PROTOCOL:
        lines = ["# FP16 矩阵精度微基准", "",
                 "- 同一 GPU、checkpoint 和二进制，3 组配对进程；16 个 shape，每个边界 2 次预热、3 次测量。",
                 "- 每个样本含 20 次 GEMM，cast-inclusive 在候选中额外含 20 次转换；不是 20 个独立 trial。",
                 "- 表中是各 trial 中位数的配对改善中位数，负数代表退化；不是完整模型或 Serving 加速。",
                 "- CUDA event 区间包含提交间隙。全量设备权重回读与 FP64 全输出 oracle 在计时外。",
                 "- 未锁频，Tensor Core 使用未验证；不产生产品晋升结论。", "",
                 "| 用例 | F32 GEMM host us | F16 GEMM host us | GEMM 改善 % | 含转换改善 % | 含转换各轮改善 % |",
                 "| --- | ---: | ---: | ---: | ---: | --- |"]
        for c in summary["cases"]:
            gemm = c["comparisons"]["gemm_only"]["host_ns_per_call"]
            cast = c["comparisons"]["cast_inclusive"]["host_ns_per_call"]
            med = gemm["trial_medians"]
            gains = ", ".join(f"{100*g:.2f}" for g in cast["paired_gains"])
            lines.append(f"| {c['name']} | {statistics.median(med[PRECISION_MODES[0]])/1000:.3f} | "
                         f"{statistics.median(med[PRECISION_MODES[1]])/1000:.3f} | "
                         f"{100*gemm['median_paired_gain']:.2f} | {100*cast['median_paired_gain']:.2f} | {gains} |")
        lines += ["", f"项目 owned allocation 减少 {100*summary['memory_reduction']:.2f}%；"
                  "完整模型数值、模型主指标和 Serving 护栏尚未验收。"]
        return "\n".join(lines) + "\n"
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
    study = parser.add_mutually_exclusive_group()
    study.add_argument("--precision", action="store_true", help="生成冻结 precision 三组配对进程计划")
    study.add_argument("--gpu-kv", action="store_true", help="生成冻结分页三组配对进程计划")
    args = parser.parse_args()
    try:
        if args.schedule:
            result = dict(reports=schedule(args.precision, args.gpu_kv),
                          statistics=KV_STATISTICS if args.gpu_kv else PRECISION_STATISTICS if args.precision else STATISTICS)
        elif args.report:
            require(args.input and sha(args.input) in (INPUT_SHA256, PRECISION_INPUT_SHA256, common.KV_INPUT_SHA256),
                    "单进程复核需要冻结的 --input")
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
                          independent_trials=result["summary"]["independent_trials"], measured_samples=result["summary"]["measured_samples"],
                          artifacts=len(result["availability"]["artifacts"]))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValidationError, OSError, ValueError, KeyError, TypeError, IndexError, zipfile.BadZipFile) as error:
        print(f"CUDA 微基准证据复核失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
