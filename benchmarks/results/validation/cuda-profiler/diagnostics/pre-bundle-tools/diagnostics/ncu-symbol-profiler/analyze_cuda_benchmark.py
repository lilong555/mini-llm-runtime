"""严格复核 CUDA 模型性能证据；统计单位为独立 trial，保留全部原始样本。"""

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
import sys
import tempfile
import zipfile


INPUT_SHA256 = "f5a311a0d7c993640ba5b761844a39e70a5ae5015db3ce9dcd07c01b6ad2a6c6"
MODEL_SHA256 = "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"
LLAMA_COMMIT = "911f6cdc8ab8a530b2bee09ee61471a6f3178eeb"
BACKENDS = ("cpu8", "cpu16", "cuda")
BENCHMARK = "minillm-cuda-runtime"
PROTOCOL = dict(
    warmup=2, measured_repeats=3, clock="steady_clock", primary="host_forward_to_token_ns",
    profiler="none", device_events=False, setup="clear_and_rebuild_independent_prefix_outside_timing",
    sampling="finite_check_and_greedy_argmax_inside_timing", reference_model_resident=False,
    generation_stop="ignore_eos_fixed_32", input_digest="sha256_count_u32le_token_position_sequence_logits_i32le")
STATISTICS = dict(
    independent_trials=5, process_median_repeats=3, comparison_trial="median_of_two_process_medians_per_backend",
    aa_trial="one_independent_process_per_arm", relative_difference="100*(B/A-1)",
    noise_percent="max(5, abs(median(d_AA)) + 2 * MAD(d_AA))", comparison_noise="max(cpu_noise,cuda_noise)",
    confidence_interval="exact_paired_percentile_bootstrap_median_95", bootstrap_resamples=3125,
    inconclusive_if_noise_percent_above=10)


class ValidationError(ValueError):
    pass


def require(value, message):
    if not value:
        raise ValidationError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"JSON 字段重复：{key}")
        result[key] = value
    return result


def decode(raw):
    def invalid(value):
        raise ValidationError(f"JSON 含非有限常量：{value}")
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=invalid)


def read(path):
    return decode(Path(path).read_text(encoding="utf-8"))


def sha(path):
    state = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            state.update(block)
    return state.hexdigest()


def digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def identical(actual, expected):
    return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def relative_path(name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name
            and not name.startswith("/") and all(p not in ("", ".", "..") for p in name.split("/")),
            f"归档路径无效：{name}")
    return Path(name)


def artifact(directory, name):
    path = Path(directory)
    for part in relative_path(name).parts:
        path = path / part
        require(not path.is_symlink(), f"归档不能依赖符号链接：{name}")
    require(path.is_file(), f"缺少证据：{name}")
    return path


def schedule():
    result = []

    def add(group, trial, backends, arms):
        for position, (backend, arm) in enumerate(zip(backends, arms)):
            result.append(dict(file=f"{backend}-{group}-t{trial}-p{position}.json", backend=backend,
                               group=group, trial=trial, arm=arm, position=position, order=len(result)))

    for trial in range(5):
        for backend in BACKENDS if trial % 2 == 0 else BACKENDS[::-1]:
            arms = ("A", "B") if trial % 2 == 0 else ("B", "A")
            add(f"aa_{backend}", trial, (backend, backend), arms)
    for cpu in BACKENDS[:2]:
        for trial in range(5):
            arms = ("A", "B", "B", "A") if trial % 2 == 0 else ("B", "A", "A", "B")
            add(f"{cpu}_cuda", trial, [cpu if arm == "A" else "cuda" for arm in arms], arms)
    return result


def batches(spec, length, sequence, logits):
    ids = spec["token_ids"]
    rows = [(ids[p % len(ids)], p, sequence, int(logits and p + 1 == length)) for p in range(length)]
    return [rows[p:p + 128] for p in range(0, len(rows), 128)]


def workload_plan(spec):
    result = []
    for work in spec["workloads"]:
        mode = work["mode"]
        setup = []
        if mode in ("prefill", "natural_generation"):
            measured = batches(spec, work.get("tokens", work.get("prompt_tokens")), 0, True)
        else:
            mixed = mode == "mixed"
            measured = batches(spec, work["prefill_tokens"], 0, True) if mixed else [[]]
            for sequence in range(int(mixed), work.get("sequences", work.get("decode_sequences", 1)) + int(mixed)):
                length = work["prefix_tokens"]
                setup.extend(batches(spec, length, sequence, False))
                measured[0].append((spec["token_ids"][length % len(spec["token_ids"])], length, sequence, 1))
        result.append((work, setup, measured))
    return result


def input_digest(batch):
    return hashlib.sha256(struct.pack("<I", len(batch)) + b"".join(struct.pack("<4i", *row) for row in batch)).hexdigest()


def allocation_snapshot(value, expected):
    require(set(value) == {"allocation_calls", "allocations", "release_calls", "releases", "allocated_bytes"}
            and all(integer(item) for item in value.values()) and identical(value, expected),
            "项目 device allocation/free 计数变化或字段无效")


def memory_plan(runtime):
    dims = runtime["dimensions"]
    weights = runtime["weights"]
    require(isinstance(weights, list) and len(weights) == 311, "权重清单不完整")
    end = payload = 0
    seen = {}
    types = Counter()
    for record in weights:
        name = record["name"]
        require(isinstance(name, str) and name not in seen and record["device_dtype"] == "F32"
                and digest(record["effective_sha256"]), "权重身份、类型或摘要无效")
        shape = record["shape"]
        require(isinstance(shape, list) and len(shape) == 2 and all(integer(v, 1) for v in shape)
                and integer(record["offset"]) and integer(record["bytes"], 1)
                and record["bytes"] == 4 * math.prod(shape), "权重尺寸无效")
        alias = record["alias_of"]
        if alias:
            require(alias in seen, "权重别名目标缺失")
            for key in ("shape", "offset", "bytes", "effective_sha256", "source_dtype"):
                require(identical(record[key], seen[alias][key]), f"共享权重字段不符：{key}")
        else:
            require(record["offset"] == (end + 255) // 256 * 256, "权重 arena 存在重叠或空洞")
            end = record["offset"] + record["bytes"]
            payload += record["bytes"]
            types[record["source_dtype"]] += 1
        seen[name] = record
    require(types == {"Q8_0": 197, "F32": 113} and seen["output.weight"]["alias_of"] == "token_embd.weight",
            "固定模型的权重 dtype 或 tied output 不符")
    embedding, q, kv, ffn = dims["embedding"], dims["heads"] * dims["head_dim"], dims["kv_heads"] * dims["head_dim"], dims["feed_forward"]
    categories = dict(activations_bytes=128 * 4 * (5 * embedding + 2 * q + 2 * kv + 2 * ffn),
                      attention_scratch_bytes=2 * 128 * dims["heads"] * 2048 * 4,
                      logits_bytes=128 * dims["vocabulary"] * 4, metadata_bytes=(7 * 128 + 4 + 2) * 4,
                      rope_bytes=2048 * dims["head_dim"] * 4)
    # 与项目 arena 的 256 字节对齐一致；各 workspace 区域按实际类型尺寸求和。
    widths = [embedding, embedding, q, kv, kv, q, embedding, ffn, ffn, embedding, embedding,
              dims["heads"] * 2048, dims["heads"] * 2048, dims["vocabulary"], 1, 1, 1, 1]
    sizes = [128 * width * 4 for width in widths] + [4 * 4, 128 * 3 * 4, 2 * 4, 2048 * dims["head_dim"] * 4]
    workspace_end = 0
    for size in sizes:
        workspace_end = (workspace_end + 255) // 256 * 256 + size
    weight_bytes = (end + 255) // 256 * 256
    workspace_bytes = (workspace_end + 255) // 256 * 256
    kv_bytes = 4 * dims["layers"] * 2 * 2048 * kv * 2
    plan = dict(weights_bytes=weight_bytes, workspace_bytes=workspace_bytes, kv_bytes=kv_bytes,
                library_workspace_bytes=4 * 1024 * 1024, **categories,
                padding_bytes=weight_bytes - payload + workspace_bytes - sum(categories.values()))
    plan["total_owned_bytes"] = weight_bytes + workspace_bytes + kv_bytes + plan["library_workspace_bytes"]
    return plan, payload, dict(types)


class State:
    def __init__(self, report):
        self.backend = report["backend"]
        self.runtime = report["runtime"]
        self.inputs = self.logits = self.calls = 0
        self.lengths = [0] * 4
        self.resident_pages = 0
        zero = dict(allocation_calls=0, allocations=0, release_calls=0, releases=0, allocated_bytes=0)
        allocation_snapshot(report["before_initialization_allocations"], zero)
        self.initial = report["initial_state"]
        self.allocations = zero
        self.plan = None
        if self.backend == "cuda":
            self.plan, self.weight_payload, types = memory_plan(self.runtime)
            require(identical(self.runtime["arithmetic"], dict(
                source_weight_dtype="Q8_0", source_tensor_counts=types, device_weight_dtype="F32",
                activation_dtype="F32", kv_dtype="F16", kv_rounding="nearest_even", qk_pv_accumulation_dtype="F32",
                softmax_exponential_dtype="F32", softmax_denominator_dtype="F64",
                gemm_compute="CUBLAS_COMPUTE_32F_PEDANTIC", fast_math=False)), "CUDA 算术模式不符")
            self.allocations = dict(zero, allocation_calls=4, allocations=4, allocated_bytes=self.plan["total_owned_bytes"])
        else:
            require(identical(self.runtime["arithmetic"], dict(source_weight_dtype="Q8_0", effective_weight_dtype="F32",
                activation_dtype="F32", kv_dtype="F16", softmax_denominator_dtype="F64", kernel="auto")), "CPU 算术模式不符")
        self.check(self.initial)

    def check(self, value):
        allocation_snapshot(value["allocation_stats"], self.allocations)
        if self.backend != "cuda":
            dims = self.runtime["dimensions"]
            pages = sum((length + 15) // 16 for length in self.lengths)
            resident = self.resident_pages * 16 * dims["layers"] * 2 * dims["kv_heads"] * dims["head_dim"] * 2
            require(value["cuda"] is None and identical(value["cpu"], dict(used_kv_pages=pages, resident_kv_bytes=resident)),
                    "CPU 独立 KV 页数或 resident payload 不符")
            return
        require(value["cpu"] is None, "CUDA 进程不能包含 CPU Runtime")
        actual = value["cuda"]
        expected = dict(state="ready", sequence_lengths=self.lengths, live_sequences=sum(v > 0 for v in self.lengths),
            live_kv_tokens=sum(self.lengths), kv_capacity_tokens=8192, resident=self.plan,
            weight_h2d_bytes=self.weight_payload, rope_h2d_bytes=self.plan["rope_bytes"],
            metadata_h2d_bytes=self.inputs * 12 + self.logits * 4, token_d2h_bytes=self.logits * 4,
            status_d2h_bytes=self.calls * 8, debug_d2h_bytes=0, intermediate_h2d_bytes=0, intermediate_d2h_bytes=0,
            owned_device_allocations=4, owned_device_bytes=self.plan["total_owned_bytes"],
            completed_forwards=self.calls, post_launch_failures=0)
        for key in ("model_load_ns", "storage_initialization_ns", "weight_decode_upload_ns"):
            expected[key] = self.runtime["initialization"][key]
            require(integer(expected[key], 1), f"CUDA 初始化时钟无效：{key}")
        require(identical(actual, expected), "CUDA KV、传输、分配或初始化状态不符")

    def call(self, record, batch, phase):
        require(record["phase"] == phase and record["input_sha256"] == input_digest(batch), "输入或执行阶段身份不符")
        require(identical(record["input_tokens"], len(batch)) and identical(record["logits_rows"], sum(r[3] for r in batch)),
                "batch token 或 logits 行数不符")
        require(identical(record["context_before"], self.lengths), "context_before 不符")
        for _, position, sequence, _ in batch:
            require(position == self.lengths[sequence] and position < 2048, "描述符不是连续独立 KV")
            self.lengths[sequence] += 1
        self.resident_pages = max(self.resident_pages, sum((length + 15) // 16 for length in self.lengths))
        require(identical(record["context_after"], self.lengths), "context_after 不符")
        require(integer(record["host_forward_to_token_ns"], 1) and record["device_elapsed_ms"] is None,
                "主时钟必须为正整数 ns，正式模型采集不能开启 events")
        samples = record["samples"]
        chosen = [(i, r[2]) for i, r in enumerate(batch) if r[3]]
        require(isinstance(samples, list) and len(samples) == len(chosen), "输出 token 行数不符")
        for sample, (index, sequence) in zip(samples, chosen):
            require(identical(sample["input_index"], index) and identical(sample["sequence"], sequence)
                    and integer(sample["token"]) and sample["token"] < self.runtime["dimensions"]["vocabulary"],
                    "输出 token 顺序或范围无效")
        self.inputs += len(batch)
        self.logits += len(chosen)
        self.calls += 1
        return [item["token"] for item in samples]


def validate_report(report, spec):
    require(identical(report["schema_version"], 1) and report["benchmark"] == BENCHMARK and report["status"] == "passed",
            "报告未完成或类型无效")
    require(report["backend"] in BACKENDS and report["input_sha256"] == INPUT_SHA256
            and report["model_sha256"] == MODEL_SHA256 and identical(report["protocol"], PROTOCOL), "报告配置不符")
    runtime, backend = report["runtime"], report["backend"]
    config = dict(max_sequences=4, max_model_len=2048, batch_tokens=128, context_pool_tokens=8192,
                  threads=None if backend == "cuda" else int(backend[3:]), kernel=None if backend == "cuda" else "auto",
                  effective_kernel=runtime["configuration"]["effective_kernel"], device=0 if backend == "cuda" else None,
                  streams=1 if backend == "cuda" else None, kv_layout="contiguous" if backend == "cuda" else "paged",
                  page_tokens=None if backend == "cuda" else 16)
    require(identical(runtime["configuration"], config), "Runtime 上限、线程或 KV 布局不符")
    require(config["effective_kernel"] in ((None,) if backend == "cuda" else ("scalar", "avx2-fma-f16c")),
            "CPU effective kernel 或 CUDA 隔离身份无效")
    dims = runtime["dimensions"]
    for key, expected in dict(embedding=1024, layers=28, heads=16, kv_heads=8, head_dim=128,
                              feed_forward=3072, vocabulary=151936).items():
        require(identical(dims[key], expected), f"固定模型尺寸不符：{key}")
    require(integer(dims["trained_context"], 8192), "模型训练 context 无效")
    require(integer(runtime["initialization"]["runtime_constructor_ns"], 1), "Runtime 初始化时钟无效")
    if backend != "cuda":
        require(runtime["device"] is None and runtime["weights"] is None, "CPU 进程不能构造 CUDA 权重或设备")
        require(all(runtime["initialization"][key] is None for key in
                    ("model_load_ns", "storage_initialization_ns", "weight_decode_upload_ns")), "CPU 初始化字段不应伪造细分计时")
    else:
        device = runtime["device"]
        require(isinstance(device["name"], str) and device["name"] and isinstance(device["uuid"], str)
                and re.fullmatch("[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", device["uuid"])
                and device["compute_capability"] == [8, 9]
                and all(integer(device[key], 1) for key in ("driver_version", "runtime_version", "cublas_version")),
                "CUDA 设备身份或工具版本无效")
    state = State(report)
    plan = workload_plan(spec)
    require(isinstance(report["workloads"], list) and len(report["workloads"]) == len(plan), "缺少或多出 workload")
    values = {}
    for actual, (work, setup, measured) in zip(report["workloads"], plan):
        name, mode = work["name"], work["mode"]
        require(actual["name"] == name and actual["mode"] == mode and actual["status"] == "passed", "workload 顺序或状态不符")
        require(isinstance(actual["iterations"], list) and len(actual["iterations"]) == 5, "必须保留 2 次 warmup 和 3 次测量")
        series = dict(samples=[], prefill_samples=[], decode_samples=[], token_ids=None)
        for i, iteration in enumerate(actual["iterations"]):
            require(identical(iteration["index"], i) and iteration["phase"] == ("warmup" if i < 2 else "measured")
                    and identical(iteration["clear_sequences"], [0, 1, 2, 3]) and integer(iteration["clear_ns"]),
                    "重复次数、warmup 或清理顺序不符")
            state.lengths = [0] * 4
            state.check(iteration["after_clear"])
            require(len(iteration["setup"]) == len(setup), "每轮必须独立重建完整 prefix")
            for call, batch in zip(iteration["setup"], setup):
                state.call(call, batch, "setup")
            state.check(iteration["before_measured"])
            phase = "prefill" if mode in ("prefill", "natural_generation") else "mixed" if mode == "mixed" else "decode"
            generated = mode == "natural_generation"
            calls = iteration["forwards"]
            require(len(calls) == len(measured) + (31 if generated else 0), "forward 次数不符")
            tokens = []
            for call, batch in zip(calls, measured):
                tokens += state.call(call, batch, phase)
            if generated:
                for step in range(1, 32):
                    batch = [(tokens[-1], work["prompt_tokens"] + step - 1, 0, 1)]
                    tokens += state.call(calls[len(measured) + step - 1], batch, "decode")
            state.check(iteration["after_measured"])
            require(identical(iteration["token_ids"], tokens), "token 序列与 forward 不符")
            if series["token_ids"] is None:
                series["token_ids"] = tokens
            require(series["token_ids"] == tokens, "同一进程的重复 greedy 输出不一致")
            sums = dict(setup_forward_ns=sum(c["host_forward_to_token_ns"] for c in iteration["setup"]),
                        host_forward_to_token_ns=sum(c["host_forward_to_token_ns"] for c in calls),
                        prefill_forward_ns=sum(c["host_forward_to_token_ns"] for c in calls if c["phase"] == "prefill"),
                        decode_forward_ns=sum(c["host_forward_to_token_ns"] for c in calls if c["phase"] == "decode"))
            for key, expected in sums.items():
                require(integer(iteration[key]) and iteration[key] == expected, f"计时汇总不符：{key}")
            if i >= 2:
                series["samples"].append(sums["host_forward_to_token_ns"])
                series["prefill_samples"].append(sums["prefill_forward_ns"])
                series["decode_samples"].append(sums["decode_forward_ns"])
        series["median_ns"] = statistics.median(series["samples"])
        values[name] = series
    state.lengths = [0] * 4
    state.check(report["final_state"])
    return dict(cases=values, forwards=state.calls, input_tokens=state.inputs, logits_rows=state.logits,
                owned_device_bytes=state.plan["total_owned_bytes"] if state.plan else 0)


def percentile(values, fraction):
    rank = fraction * (len(values) - 1)
    low, high = math.floor(rank), math.ceil(rank)
    return values[low] + (values[high] - values[low]) * (rank - low)


def paired_statistics(a, b):
    require(len(a) == len(b) == 5 and all(finite(v) and v > 0 for v in a + b), "统计必须使用 5 个独立 trial 的正数中位值")
    differences = [100 * (right - left) / left for left, right in zip(a, b)]
    center = statistics.median(differences)
    mad = statistics.median(abs(value - center) for value in differences)
    # 枚举 5^5 个成对 trial 重采样，避免随机种子或重复样本被误当独立实验。
    boot = sorted(statistics.median(sample) for sample in itertools.product(differences, repeat=5))
    return dict(a_trial_median_ns=a, b_trial_median_ns=b, paired_relative_percent=differences,
                median_relative_percent=center, mad_percent=mad,
                paired_ci95_percent=[percentile(boot, .025), percentile(boot, .975)],
                median_paired_speedup=statistics.median(left / right for left, right in zip(a, b)))


def classify(stats, noise):
    if noise > 10:
        return "measurement_inconclusive"
    low, high = stats["paired_ci95_percent"]
    if abs(stats["median_relative_percent"]) <= noise or low <= 0 <= high:
        return "inconclusive"
    return "faster" if stats["median_relative_percent"] < 0 else "slower"


def analyze_pairs(reports, spec):
    expected = schedule()
    require(len(reports) == len(expected) and all(identical(r["process"], slot) for r, slot in zip(reports, expected)),
            "A/A、trial 或平衡进程顺序不完整")
    audited = [validate_report(report, spec) for report in reports]
    tokens = {}
    for slot, result in zip(expected, audited):
        for name, case in result["cases"].items():
            key = (slot["backend"], name)
            tokens.setdefault(key, case["token_ids"])
            require(identical(case["token_ids"], tokens[key]),
                    f"同一 backend 的跨进程输出不同：{slot['backend']}/{name}，报告 {slot['file']}")
    aa, comparisons = {}, {}
    for backend in BACKENDS:
        aa[backend] = {}
        for work in spec["workloads"]:
            name = work["name"]
            a, b = [], []
            for trial in range(5):
                for arm, target in (("A", a), ("B", b)):
                    values = [result["cases"][name]["median_ns"] for slot, result in zip(expected, audited)
                              if slot["group"] == f"aa_{backend}" and slot["trial"] == trial and slot["arm"] == arm]
                    require(len(values) == 1, "A/A trial 不是独立进程对")
                    target.append(values[0])
            stats = paired_statistics(a, b)
            stats["noise_percent"] = max(5.0, abs(stats["median_relative_percent"]) + 2 * stats["mad_percent"])
            stats["status"] = "measurement_inconclusive" if stats["noise_percent"] > 10 else "measured"
            aa[backend][name] = stats
    for cpu in BACKENDS[:2]:
        comparisons[cpu] = {}
        for work in spec["workloads"]:
            name, a, b = work["name"], [], []
            agreement = True
            token_pairs = []
            for trial in range(5):
                sides = {}
                for arm, target in (("A", a), ("B", b)):
                    values = [result["cases"][name] for slot, result in zip(expected, audited)
                              if slot["group"] == f"{cpu}_cuda" and slot["trial"] == trial and slot["arm"] == arm]
                    require(len(values) == 2 and values[0]["token_ids"] == values[1]["token_ids"],
                            "同一 backend 的配对进程输出不同")
                    target.append(statistics.median(v["median_ns"] for v in values))
                    sides[arm] = values[0]["token_ids"]
                token_pairs.append(sides)
                agreement = agreement and sides["A"] == sides["B"]
            stats = paired_statistics(a, b)
            noise = max(aa[cpu][name]["noise_percent"], aa["cuda"][name]["noise_percent"])
            stats.update(noise_percent=noise, status=classify(stats, noise),
                         token_agreement=agreement, trial_tokens=token_pairs)
            if not agreement:
                stats["status"] = "correctness_followup_required"
            comparisons[cpu][name] = stats
    all_stats = [v for group in comparisons.values() for v in group.values()]
    overall = "correctness_followup_required" if any(not v["token_agreement"] for v in all_stats) else (
        "measurement_inconclusive" if any(v["status"] == "measurement_inconclusive" for v in all_stats) else "measured")
    return dict(schema_version=1, benchmark=BENCHMARK, scope="CUDA-VS-001 Step 8 model performance",
                status=overall, statistics=STATISTICS, reports=len(reports), independent_trials=5,
                measured_repetitions=sum(len(v["samples"]) for r in audited for v in r["cases"].values()),
                data_path_gates="passed", aa=aa, comparisons=comparisons,
                microbenchmark_complete=False, profiler_complete=False, gpu_serving=False)


def source_archive(directory, source):
    state_file = artifact(directory, source["state_file"])
    archive_file = artifact(directory, source["snapshot"]["path"])
    require(sha(state_file) == source["worktree_state_sha256"] and sha(archive_file) == source["snapshot"]["sha256"],
            "源码清单或 ZIP 摘要不符")
    state = read(state_file)
    require(identical(state["scope"], source["scope"]) and state["files"], "源码 scope 不符或为空")
    files = {}
    with zipfile.ZipFile(archive_file) as archive:
        require(len(archive.infolist()) == len(state["files"]), "源码 ZIP 项数不符")
        require(len(set(archive.namelist())) == len(archive.namelist()), "源码 ZIP 包含重复路径")
        for record in state["files"]:
            name = record["path"]
            relative_path(name)
            require(name not in files and digest(record["sha256"]) and integer(record["size_bytes"]), "源码记录无效")
            info = archive.getinfo(name)
            require(not info.is_dir() and (info.external_attr >> 16) & 0o170000 != 0o120000
                    and info.file_size == record["size_bytes"], f"源码尺寸或类型不符：{name}")
            raw = archive.read(name)
            require(hashlib.sha256(raw).hexdigest() == record["sha256"], f"源码内容不符：{name}")
            files[name] = record
        require("apps/cuda_runtime_bench.cpp" in files and "src/minillm/cuda/runtime.cpp" in files
                and "scripts/analyze_cuda_benchmark.py" in files, "源码 ZIP 缺少基准或 Runtime 源码")
        require(files["benchmarks/runtime-inputs/qwen3-cuda-v0.json"]["sha256"] == INPUT_SHA256, "源码中的冻结输入不符")
    return files


def validate_bundle(directory):
    manifest_path = artifact(directory, "manifest.json")
    manifest = read(manifest_path)
    require(identical(manifest["schema_version"], 1) and manifest["benchmark"] == BENCHMARK
            and identical(manifest["reports"], schedule()) and identical(manifest["statistics"], STATISTICS),
            "manifest 类型、完整进程计划或统计协议不符")
    require(manifest["dependencies"]["llama_commit"] == LLAMA_COMMIT
            and manifest["build"]["own_cuda"] == "ON" and manifest["build"]["upstream_cuda"] == "OFF"
            and manifest["build"]["type"] in ("Release", "RelWithDebInfo"), "构建边界或依赖不符")
    require(manifest["model"]["sha256"] == MODEL_SHA256 and digest(manifest["binary"]["sha256"])
            and manifest["input"]["sha256"] == INPUT_SHA256 and manifest["protocol_id"] == "qwen3-cuda-model-v0",
            "模型、二进制或协议身份不符")
    files = source_archive(directory, manifest["source"])
    spec_path = artifact(directory, manifest["input"]["path"])
    require(sha(spec_path) == INPUT_SHA256, "冻结输入摘要不符")
    spec = read(spec_path)
    collection = read(artifact(directory, "collection-status.json"))
    require(collection["status"] == "passed" and len(collection["reports"]) == 70, "采集不完整或曾报告执行失败")
    availability = []
    seen = set()

    def check(name, expected=None):
        require(name not in seen, f"重复 artifact：{name}")
        path = artifact(directory, name)
        actual = sha(path)
        require(expected is None or digest(expected) and actual == expected, f"artifact 摘要不符：{name}")
        seen.add(name)
        availability.append(dict(locator=name, mandatory=True, exists=True, sha256=actual,
                                 size_bytes=path.stat().st_size, status="AVAILABLE"))
        return path

    for name in ("manifest.json", "collection-status.json", manifest["input"]["path"],
                 manifest["source"]["state_file"], manifest["source"]["snapshot"]["path"]):
        check(name)
    for item in manifest["artifacts"]:
        check(item["path"], item["sha256"])
    require({"validation-summary.json", "numerical-source-state.json", "numerical-environment.json", "verify.py"} <= seen,
            "缺少数值摘要、编译身份、源码继承或复核入口")
    require(sha(artifact(directory, "verify.py")) == files["scripts/analyze_cuda_benchmark.py"]["sha256"],
            "归档复核入口不属于采集源码")
    numeric = read(artifact(directory, "validation-summary.json"))
    numeric_environment = read(artifact(directory, "numerical-environment.json"))
    validation_binary = manifest["numerical_evidence"]["validation_binary_sha256"]
    require(digest(validation_binary) and validation_binary == numeric_environment["full_validation_binary_sha256"],
            "当前完整模型测试的二进制不属于既有数值验收，必须重新验证")
    require(numeric["status"] == "passed" and numeric["complete"] is True and numeric["passed"] is True
            and numeric["full_corpus_contract"] is True and numeric["model_sha256"] == MODEL_SHA256
            and numeric["totals"]["numeric_failures"] == 0, "完整数值门禁未通过")
    old_files = {item["path"]: item for item in read(artifact(directory, "numerical-source-state.json"))["files"]}
    product = [name for name in files if name.startswith(("include/minillm/", "src/minillm/"))]
    old_product = {name for name in old_files if name.startswith(("include/minillm/", "src/minillm/"))}
    require(product and set(product) == old_product
            and all(files[name]["sha256"] == old_files[name]["sha256"] for name in product),
            "数学或 Runtime 源码不同，不能沿用既有数值门禁")
    reports = []
    for slot, record in zip(manifest["reports"], collection["reports"]):
        require(record["file"] == slot["file"] and record["exit_code"] == 0, "采集执行状态或顺序不符")
        report = read(check(record["file"], record["sha256"]))
        require(report["scope"] == "paired_model_baseline" and identical(report["process"], slot), "报告进程身份不符")
        require(identical(report["run_identity"], dict(run_id=manifest["run_id"], manifest_sha256=sha(manifest_path),
            source_state_sha256=manifest["source"]["worktree_state_sha256"], binary_sha256=manifest["binary"]["sha256"])),
            "报告未绑定本次源码、二进制或 manifest")
        for item in record["artifacts"]:
            check(item["path"], item["sha256"])
        require({slot["file"] + ".stdout.txt", slot["file"] + ".stderr.txt", slot["file"] + ".process.json"} <= seen,
                "缺少完整 stdout、stderr 或进程环境")
        process = read(artifact(directory, slot["file"] + ".process.json"))
        require(process["order"] == slot["order"] and process["exit_code"] == 0
                and "before" in process and "after" in process and process["arguments"], "进程环境或原始命令缺失")
        reports.append(report)
    devices = [r["runtime"]["device"] for r in reports if r["backend"] == "cuda"]
    require(devices and all(identical(d, devices[0]) for d in devices), "采集中 GPU 或 CUDA 版本发生变化")
    for backend in BACKENDS[:2]:
        effective = {r["runtime"]["configuration"]["effective_kernel"] for r in reports if r["backend"] == backend}
        require(len(effective) == 1, "CPU SIMD 配置发生变化")
    summary = analyze_pairs(reports, spec)
    summary.update(run_id=manifest["run_id"], source_state_sha256=manifest["source"]["worktree_state_sha256"])
    cuda = next(r for r in reports if r["backend"] == "cuda")
    copy = []
    for report in reports:
        state = report["final_state"]
        d = state["cuda"]
        copy.append(dict(file=report["process"]["file"], backend=report["backend"],
                         steady_project_allocation_calls=0, steady_project_release_calls=0,
                         steady_weight_h2d_bytes=0, steady_hidden_h2d_d2h_bytes=0,
                         metadata_h2d_bytes=d["metadata_h2d_bytes"] if d else None,
                         token_d2h_bytes=d["token_d2h_bytes"] if d else None,
                         status_d2h_bytes=d["status_d2h_bytes"] if d else None,
                         debug_d2h_bytes=d["debug_d2h_bytes"] if d else None,
                         owned_device_bytes=d["owned_device_bytes"] if d else 0))
    return dict(summary=summary, availability=dict(schema_version=1, status="AVAILABLE", artifacts=availability),
                copies=dict(schema_version=1, status="passed", scope="project_owned_not_vendor_internal", reports=copy),
                weights=cuda["runtime"]["weights"], memory=cuda["initial_state"]["cuda"]["resident"])


def analysis_text(summary):
    lines = ["# 自有 CUDA 模型性能基线", "", f"- 状态：`{summary['status']}`。",
             "- 主指标为无 profiler 的 host forward 到 token 延迟，包含必要 copy、finite/argmax 和同步。",
             "- 每个配置使用 5 个独立 trial；每进程 2 次 warmup、3 次测量。表中负百分比表示 CUDA 更快。",
             "- A/A 噪声取 CPU 与 CUDA 的较大值；超过 10% 时不可据此宣称加速或没有退化。",
             "- 95% 区间为 5^5 次成对 trial 重采样的 percentile bootstrap；仅 5 个 trial，区间稳定性有限。",
             "- 数值门禁继承同一 Runtime 源码的完整语料验收；本包保留摘要及源码对应，不重复常驻数值 reference。",
             "- 本报告仅覆盖模型层，不代表 microbenchmark、Profiler、HTTP TTFT、GPU Serving 或 PagedAttention。", "",
             "| CPU 对照 | 用例 | CPU ms | CUDA ms | 配对差异 % | 95% 区间 % | 噪声 % | 结论 |",
             "| --- | --- | ---: | ---: | ---: | --- | ---: | --- |"]
    for cpu, cases in summary["comparisons"].items():
        for name, value in cases.items():
            a = statistics.median(value["a_trial_median_ns"]) / 1e6
            b = statistics.median(value["b_trial_median_ns"]) / 1e6
            low, high = value["paired_ci95_percent"]
            lines.append(f"| {cpu} | {name} | {a:.3f} | {b:.3f} | {value['median_relative_percent']:.2f} | "
                         f"[{low:.2f}, {high:.2f}] | {value['noise_percent']:.2f} | {value['status']} |")
    return "\n".join(lines) + "\n"


def publish(directory, outputs):
    root = Path(directory)
    with tempfile.TemporaryDirectory(prefix=".analysis-", dir=root) as temporary:
        stage = Path(temporary)
        originals, changed = {}, []
        for name, value in outputs.items():
            require(relative_path(name).name == name, "分析输出必须位于归档根目录")
            target = root / name
            require(not target.is_symlink() and (not target.exists() or target.is_file()), "分析输出路径不是普通文件")
            originals[name] = target.read_bytes() if target.exists() else None
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            (stage / name).write_text(text, encoding="utf-8")
        try:
            for name in outputs:
                os.replace(stage / name, root / name)
                changed.append(name)
        except OSError:
            for name in reversed(changed):
                if originals[name] is None:
                    (root / name).unlink()
                else:
                    (stage / name).write_bytes(originals[name])
                    os.replace(stage / name, root / name)
            raise


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
            result = validate_report(read(args.report), read(args.input))
        else:
            result = validate_bundle(args.directory)
            if args.write:
                publish(args.directory, {"availability.json": result["availability"], "weight-plan.json": result["weights"],
                    "memory-plan.json": result["memory"], "copy-allocation-summary.json": result["copies"],
                    "analysis.md": analysis_text(result["summary"]), "summary.json": result["summary"]})
            result = dict(status=result["summary"]["status"], reports=result["summary"]["reports"],
                          data_path_gates=result["summary"]["data_path_gates"],
                          artifacts=len(result["availability"]["artifacts"]))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValidationError, OSError, ValueError, KeyError, TypeError, IndexError, zipfile.BadZipFile) as error:
        print(f"CUDA 性能证据复核失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
