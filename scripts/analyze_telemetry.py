#!/usr/bin/env python3
"""验收在线 batch 与 SSE token 关联，并汇总经过时间；仅使用标准库。"""

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, f"非法非负整数：{value!r}")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"重复 JSON 字段：{key}")
        result[key] = value
    return result


def parse(text):
    return json.loads(text, object_pairs_hook=unique_object,
                      parse_constant=lambda value: require(False, f"非有限值：{value}"))


def read(path):
    return parse(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    def percentile(q):
        rank = q * (len(ordered) - 1)
        low, high = math.floor(rank), math.ceil(rank)
        return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)
    return {"count": len(ordered), "min": ordered[0], "max": ordered[-1],
            "p50": percentile(.5), "p95": percentile(.95), "p99": percentile(.99)}


def check_resources(value, backend, version, engine):
    if backend not in ("minillm", "minillm-cuda"):
        require(value is None, "不支持的后端不能伪报物理 KV 数据")
        return
    require(type(value) is dict, "MiniLLM 物理 KV 数据缺失")
    integer(value["resident_kv_payload_bytes"])
    if version == 1:
        require(backend == "minillm", "旧 schema 不支持 own-CUDA 资源")
        integer(value["live_kv_pages"])
        return
    require(value["state_valid"] is True and value["reusable"] is True, "成功时间线包含无效或隔离资源")
    if backend == "minillm-cuda":
        require(value["layout"] == "contiguous" and value["live_kv_pages"] is None, "连续 GPU KV 不提供页数")
        require(integer(value["capacity_tokens"], 1) == engine["max_active"] * engine["max_model_len"], "GPU 物理槽容量错误")
        require(integer(value["live_tokens"]) <= value["capacity_tokens"], "GPU live tokens 越界")
        require(integer(value["owned_device_bytes"], 1) >= integer(value["resident_kv_payload_bytes"], 1), "GPU resident/owned 字节数非法")
    else:
        require(value["layout"] == "paged", "CPU KV 布局错误")
        integer(value["live_kv_pages"])
        require(value["capacity_tokens"] == engine["context_tokens"], "CPU KV 容量错误")
        require(value["live_tokens"] is None and value["owned_device_bytes"] is None, "CPU 不提供未测量的 token/device 数量")


def validate_capture(rows, report, engine):
    require(len(rows) >= 2, "缺失采集头或终态")
    header, footer, batches = rows[0], rows[-1], rows[1:-1]
    version = header["schema_version"]
    require(header["type"] == "header" and type(version) is int and version in (1, 2), "采集头不合法")
    require(header["clock"] == "engine_relative_steady_ns", "计时口径不兼容")
    require(header["runtime_replay_available"] is False, "不支持模型重放格式")
    require(header["mode"] == engine["telemetry_mode"] and header["mode"] in ("batches", "stages"), "观测模式不同")
    require(header["backend"] == engine["metrics_backend"], "后端身份不同")
    require(header["backend"] in ("minillm", "minillm-cuda", "llama.cpp"), "未知后端")
    if version == 2:
        require(header["resource_boundaries"] == {
            "before": "before_execute", "after": "after_execute_before_request_cleanup",
            "final": "after_engine_stop_before_runner_destruction"}, "资源快照边界不明确")
        caps = header["capabilities"]
        for key, expected in (("max_sequences", engine["max_active"] + engine.get("prefix_cache_entries", 0)),
                              ("max_batch_tokens", engine["batch_tokens"]), ("max_model_len", engine["max_model_len"])):
            require(integer(caps[key]) == 0 or caps[key] >= expected, "后端容量不足")
        require(caps["synchronous_execute"] is True, "时间线需要同步完成边界")
        require(caps["runtime_stage_profile"] is (header["backend"] == "minillm"), "阶段测量能力不真实")
        require(caps["prefix_copy"] is (header["backend"] != "minillm-cuda"), "prefix 能力错误")
    if header["backend"] == "minillm-cuda":
        require(version == 2, "own-CUDA 需要 schema v2")
        require(caps["max_sequences"] == engine["max_active"] and caps["max_batch_tokens"] == engine["batch_tokens"] and
                caps["max_model_len"] == engine["max_model_len"], "own-CUDA 能力不是实例实际容量")
        require(1 <= engine["max_active"] <= 4 and 1 <= engine["batch_tokens"] <= 128 and
                2 <= engine["max_model_len"] <= 2048, "own-CUDA 容量超限")
        require(engine.get("prefix_cache_entries", 0) == engine.get("prefix_cache_tokens", 0) == 0, "own-CUDA 不支持 prefix")
        require(engine["context_tokens"] <= engine["max_active"] * engine["max_model_len"], "GPU 容量信用超过物理槽")
    require(header["policy"] == report["server_before"]["policy"], "策略身份不同")
    require(integer(header["capacity"], 1) == engine["telemetry_capacity"], "缓冲容量不同")
    integer(header["storage_bytes"], 1)
    require(footer["type"] == "footer" and footer["complete"] is True, "采集未完整结束")
    require(integer(footer["dropped"]) == 0 and footer["engine_error"] == "", "采集丢失或模型执行失败")
    require(integer(footer["recorded"]) == len(batches) <= header["capacity"], "记录数不完整")
    require(len(batches) == report["server_after"]["batches"], "采集和服务 batch 数不同")
    check_resources(footer["resources_final"], header["backend"], version, engine)
    if footer["resources_final"] is not None:
        live_field = "live_tokens" if header["backend"] == "minillm-cuda" else "live_kv_pages"
        require(footer["resources_final"][live_field] == 0, "停服后仍有活跃 KV")
    token_map, request_slices, seen_orders = {}, collections.defaultdict(list), {}
    previous_finish = 0
    measured = []
    for index, batch in enumerate(batches, 1):
        require(batch["type"] == "batch" and integer(batch["batch_id"], 1) == index, "缺失或重复 batch")
        require(batch["completed"] is True and batch["runner_completed"] is True, "不完整 batch")
        for key in ("start_ns", "admission_ns", "scheduler_ns", "prepare_ns", "runner_start_ns",
                    "runner_ns", "finish_ns", "prefill_tokens", "decode_tokens", "logits_tokens",
                    "waiting_requests", "active_requests", "sequences", "context_before_sum",
                    "context_before_max", "context_after_sum", "context_after_max", "reserved_unique_blocks"):
            integer(batch[key])
        require(batch["start_ns"] >= previous_finish, "batch 时间逆序")
        require(batch["start_ns"] + batch["admission_ns"] + batch["scheduler_ns"] + batch["prepare_ns"] == batch["runner_start_ns"], "batch 前置时间不守恒")
        require(batch["runner_start_ns"] + batch["runner_ns"] <= batch["finish_ns"], "runner 时间越界")
        previous_finish = batch["finish_ns"]
        require(0 < batch["sequences"] == len(batch["slices"]) <= batch["active_requests"] <= engine["max_active"], "序列数非法")
        require(batch["waiting_requests"] + batch["active_requests"] <= engine["queue_capacity"], "请求数超出容量")
        require(batch["reserved_unique_blocks"] <= engine["context_tokens"] // engine["block_size"], "容量信用越界")
        for field in ("resources_before", "resources_after"):
            check_resources(batch[field], header["backend"], version, engine)
            if batch[field] is not None and batch[field]["live_kv_pages"] is not None:
                require(batch[field]["live_kv_pages"] <= engine["context_tokens"] // engine["block_size"], "物理页越界")
            if header["backend"] == "minillm-cuda":
                for key in ("resident_kv_payload_bytes", "capacity_tokens", "owned_device_bytes"):
                    require(batch[field][key] == footer["resources_final"][key], "GPU 常驻分配或容量发生变化")
        if header["backend"] == "minillm-cuda":
            require(batch["resources_after"]["live_tokens"] == batch["resources_before"]["live_tokens"] +
                    batch["prefill_tokens"] + batch["decode_tokens"], "GPU committed token 数量不守恒")
        counts = collections.Counter()
        contexts, after_contexts, sequences, orders = [], [], set(), set()
        for item in batch["slices"]:
            require(type(item["prefill"]) is bool and type(item["emitted"]) is bool, "slice 布尔字段非法")
            sequence = integer(item["sequence"])
            order = integer(item["request_order"], 1)
            require(sequence < engine["max_active"] and sequence not in sequences and order not in orders, "重复或非法 sequence/request")
            sequences.add(sequence)
            orders.add(order)
            require(type(item["request_id"]) is str and item["request_id"], "请求标识缺失")
            require(seen_orders.setdefault(order, item["request_id"]) == item["request_id"], "request_order 复用")
            count = integer(item["tokens"], 1)
            context = integer(item["context_before"])
            require(context + count <= engine["max_model_len"], "请求上下文越界")
            integer(item["token_index"])
            logits = integer(item["logits_tokens"])
            require(logits <= 1 and (item["prefill"] or (count == 1 and logits == 1)), "decode/logits 数量非法")
            counts["prefill_tokens" if item["prefill"] else "decode_tokens"] += count
            counts["logits_tokens"] += logits
            contexts.append(context)
            after_contexts.append(context + count)
            request_slices[(item["request_id"], order)].append(item)
            if logits:
                integer(item["sampled_token"])
            else:
                require(item["sampled_token"] is None and not item["emitted"], "无 logits 的 slice 产生输出")
            integer(item["emitted_ns"])
            if item["emitted"]:
                require(batch["runner_start_ns"] + batch["runner_ns"] <= item["emitted_ns"] <= batch["finish_ns"], "token 时间越界")
                key = (item["request_id"], order, item["token_index"])
                require(key not in token_map, "重复输出 token")
                token_map[key] = (index, item["sampled_token"], item["emitted_ns"])
        require(all(batch[key] == counts[key] for key in ("prefill_tokens", "decode_tokens", "logits_tokens")), "batch token 汇总不一致")
        require(batch["prefill_tokens"] + batch["decode_tokens"] <= engine["batch_tokens"], "batch 超过预算")
        require(batch["context_before_sum"] == sum(contexts) and batch["context_before_max"] == max(contexts), "前置 KV 长度汇总不一致")
        require(batch["context_after_sum"] == sum(after_contexts) and batch["context_after_max"] == max(after_contexts), "后置 KV 长度汇总不一致")
        profile = batch["runner"]
        if header["mode"] == "stages" and header["backend"] == "minillm":
            require(type(profile) is dict and profile["completed"] is True, "缺少 Runtime 阶段")
            stages = profile["stages"]
            expected = {"kv_prepare", "embedding", "rope_prepare", "attention_norm", "query_projection",
                        "key_projection", "value_projection", "qk_norm_rope_kv", "attention", "output_projection",
                        "attention_residual", "ffn_norm", "gate_projection", "up_projection", "swiglu",
                        "down_projection", "ffn_residual"}
            if batch["logits_tokens"]:
                expected |= {"final_norm", "lm_head"}
            require({s["name"] for s in stages} == expected and len(stages) == len(expected), "阶段集合缺失或重复")
            for stage in stages:
                for key in ("calls", "wall_ns", "matrix_m", "matrix_n", "matrix_k", "parallel_wall_ns", "caller_wait_ns", "worker_work_sum_ns"):
                    integer(stage[key])
                require(stage["calls"] > 0 and stage["varying_shape"] is False, "阶段形状或调用数非法")
                require(stage["caller_wait_ns"] <= stage["parallel_wall_ns"] <= stage["wall_ns"], "线程池计时越界")
                if stage["name"].endswith("projection") or stage["name"] == "lm_head":
                    expected_m = 1 if stage["name"] == "lm_head" else sum((batch["prefill_tokens"], batch["decode_tokens"]))
                    require(stage["matrix_m"] == expected_m and stage["matrix_n"] > 0 and stage["matrix_k"] > 0, "矩阵形状非法")
                else:
                    require(stage["matrix_m"] == stage["matrix_n"] == stage["matrix_k"] == 0, "非矩阵阶段含矩阵形状")
                if stage["name"] in ("final_norm", "lm_head"):
                    require(stage["calls"] == batch["logits_tokens"], "输出层调用数不同")
            require(sum(s["wall_ns"] for s in stages) + integer(profile["unaccounted_ns"]) == integer(profile["forward_ns"]), "Runtime 阶段时间不守恒")
            require(profile["forward_ns"] + integer(profile["sampling_ns"]) <= batch["runner_ns"], "Runtime 时间超过 runner")
        else:
            require(profile is None, "不支持或未启用的 Runtime 阶段应为 null")
        if index > report["server_before"]["batches"]:
            measured.append(batch)
    linked, stalls = set(), []
    require(all(r["success"] is True for r in report["requests"]), "时间线分析要求全部请求成功；失败报告保留供协议检查")
    for request in report["requests"]:
        telemetry = request["token_telemetry"]
        require(len(telemetry) == len(request["token_ids"]) > 0, "客户端 token 关联缺失")
        for index, (token, sample) in enumerate(zip(request["token_ids"], telemetry)):
            require(type(sample) is dict and integer(sample["token_index"]) == index, "客户端 token 序号非法")
            key = (request["id"], integer(sample["request_order"], 1), index)
            require(key in token_map and key not in linked, "客户端 token 无唯一 batch 记录")
            require(token_map[key] == (integer(sample["batch_id"], 1), token, integer(sample["engine_elapsed_ns"])), "客户端与 batch 输出不一致")
            require(sample["batch_id"] > report["server_before"]["batches"], "测量 token 指向预热 batch")
            linked.add(key)
        require(len({sample["request_order"] for sample in telemetry}) == 1, "同一请求的 order 不一致")
        prompt = request["usage"]["prompt_tokens"]
        position = request["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        generated = 0
        for item in request_slices[(request["id"], telemetry[0]["request_order"])]:
            require(item["context_before"] == position and item["token_index"] == generated, "请求位置或生成序号不连续")
            require(item["prefill"] == (position < prompt), "prefill/decode 分类不符")
            position += item["tokens"]
            require(not item["prefill"] or position <= prompt, "prefill 超出 prompt")
            require(item["logits_tokens"] == int(position >= prompt), "logits 请求不符")
            generated += int(item["emitted"])
        require(generated == len(telemetry), "请求 slice 输出集合不完整")
        intervals = [b - a for a, b in zip(request["token_times_ms"], request["token_times_ms"][1:])]
        server_intervals = [(b["engine_elapsed_ns"] - a["engine_elapsed_ns"]) / 1e6 for a, b in zip(telemetry, telemetry[1:])]
        stalls.append({"request_id": request["id"], "client_max_itl_ms": max(intervals, default=None),
                       "engine_max_itl_ms": max(server_intervals, default=None),
                       "interval_residual_ms": [a - b for a, b in zip(intervals, server_intervals)]})
    measured_keys = {key for key, value in token_map.items() if value[0] > report["server_before"]["batches"]}
    require(measured_keys == linked, "缺失客户端输出或存在外部请求")
    known_requests = {(key[0], key[1]) for key in linked}
    require(all((s["request_id"], s["request_order"]) in known_requests
                for b in measured for s in b["slices"]), "测量期间存在未登记请求")
    return measured, stalls


def summarize_batches(batches):
    total = sum(b["finish_ns"] - b["start_ns"] for b in batches)
    stage_ns = collections.Counter()
    composition = collections.Counter()
    for batch in batches:
        kind = "mixed" if batch["prefill_tokens"] and batch["decode_tokens"] else "prefill" if batch["prefill_tokens"] else "decode"
        composition[f"{kind}:{batch['prefill_tokens']}+{batch['decode_tokens']}"] += 1
        if batch["runner"]:
            for stage in batch["runner"]["stages"]:
                stage_ns[stage["name"]] += stage["wall_ns"]
    return {"batches": len(batches), "composition": dict(sorted(composition.items())),
            "iteration_sum_ms": total / 1e6,
            "scheduler_percent": 100 * sum(b["scheduler_ns"] for b in batches) / total if total else None,
            "admission_percent": 100 * sum(b["admission_ns"] for b in batches) / total if total else None,
            "runner_percent": 100 * sum(b["runner_ns"] for b in batches) / total if total else None,
            "runner_ms": distribution([b["runner_ns"] / 1e6 for b in batches]),
            "stage_sum_ms": {k: v / 1e6 for k, v in stage_ns.items()}}


def analyze(directories):
    records, identities, outputs = [], {}, {}
    for directory in directories:
        command = ["pwsh", "-NoProfile", "-File", str(Path(__file__).with_name("Analyze-Benchmarks.ps1")), "-Directory", str(directory)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        require(result.returncode == 0, f"Serving 归档验收失败：{directory}\n{result.stdout}\n{result.stderr}")
        manifest = read(directory / "manifest.json")
        engine = manifest["engine"]
        mode = engine.get("telemetry_mode", "off")
        fixed_engine = {k: v for k, v in engine.items() if k != "telemetry_mode"}
        fixed_protocol = {k: v for k, v in manifest["protocol"].items() if k not in ("profiler_mode", "trials_per_variant", "order_offset")}
        condition = json.dumps([manifest["trace"]["sha256"], fixed_engine, fixed_protocol], sort_keys=True)
        identity = [manifest["source"]["worktree_state_sha256"], manifest["model"]["sha256"],
                    {k: v["sha256"] for k, v in manifest["binaries"].items()}, manifest["build"],
                    manifest["dependencies"], manifest["environment"]]
        require(identities.setdefault(condition, identity) == identity, "对照源码、模型、环境或二进制身份不同")
        for spec in manifest["reports"]:
            report_path = directory / spec["file"]
            report = read(report_path)
            require(all(r["success"] is True for r in report["requests"]), "在线归因要求成功请求集合")
            tokens = {r["id"]: r["token_ids"] for r in report["requests"]}
            require(outputs.setdefault(condition, tokens) == tokens, "观测模式或轮次间输出 token 不一致")
            record = {"directory": str(directory), "file": spec["file"], "mode": mode, "policy": spec["variant"],
                      "trial": spec["trial"], "condition_sha256": hashlib.sha256(condition.encode()).hexdigest(),
                      "manifest_sha256": sha256(directory / "manifest.json"), "report_sha256": sha256(report_path),
                      "summary": report["summary"], "metrics_batch_delta": report["server_after"]["batches"] - report["server_before"]["batches"],
                      "metrics_mixed_delta": report["server_after"]["mixed_batches"] - report["server_before"]["mixed_batches"]}
            if mode != "off":
                name = spec["telemetry_file"]
                require(Path(name).name == name, "观测文件必须位于归档目录内")
                path = directory / name
                rows = [parse(line) for line in path.read_text(encoding="utf-8").splitlines()]
                batches, stalls = validate_capture(rows, report, engine)
                record.update(telemetry_sha256=sha256(path), batches=summarize_batches(batches), request_stalls=stalls)
            else:
                require(all(all(v is None for v in r["token_telemetry"]) for r in report["requests"]), "关闭观测仍返回 token 观测信息")
            records.append(record)
    comparisons = []
    for condition in sorted({r["condition_sha256"] for r in records}):
        for policy in ("mixed", "prefill_first"):
            matched = [r for r in records if r["condition_sha256"] == condition and r["policy"] == policy]
            baseline = [r for r in matched if r["mode"] == "off"]
            if not baseline:
                continue
            for mode in ("batches", "stages"):
                observed = [r for r in matched if r["mode"] == mode]
                if not observed:
                    continue
                comparison = {"condition_sha256": condition, "policy": policy, "mode": mode,
                              "off_processes": len(baseline), "on_processes": len(observed)}
                for metric in ("elapsed_s", "output_tokens_per_second"):
                    off = [r["summary"][metric] for r in baseline]
                    on = [r["summary"][metric] for r in observed]
                    comparison[metric] = {"off": distribution(off), "on": distribution(on),
                                          "median_change_percent": 100 * (statistics.median(on) / statistics.median(off) - 1)}
                comparison["batch_counts"] = {"off": [r["metrics_batch_delta"] for r in baseline], "on": [r["metrics_batch_delta"] for r in observed]}
                comparison["mixed_batch_counts"] = {"off": [r["metrics_mixed_delta"] for r in baseline], "on": [r["metrics_mixed_delta"] for r in observed]}
                comparisons.append(comparison)
    return {"schema_version": 1, "status": "passed", "reports": records, "comparisons": comparisons,
            "limits": ["独立进程延迟差异包含系统噪声与在线 batch 变化，不是纯插桩耗时。",
                       "客户端与 Engine 时钟原点不同，只比较对应 token 间隔，不将绝对时间相减。",
                       "interval_residual_ms 包含事件队列、HTTP、socket 缓冲和客户端调度，不等于网络延迟。",
                       "阶段按层求和；worker 经过时间重叠，不能计入 forward wall time。",
                       "未采集 sequence 生命周期与完整输入，不提供 Runtime replay；attention 尚未分离页查找、QK、softmax、PV。"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = analyze(args.directories)
    except (ValueError, KeyError, TypeError, OSError, IndexError, ZeroDivisionError) as error:
        summary = {"schema_version": 1, "status": "failed", "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "reports": len(summary.get("reports", [])),
                      "error": summary.get("error")}, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
