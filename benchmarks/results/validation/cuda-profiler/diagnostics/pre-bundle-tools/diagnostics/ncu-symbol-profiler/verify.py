"""完整 28 层模型的外部 Profiler 证据复核；回放或插桩时间不作为正式基线。"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import statistics
import sys
import tempfile
import zipfile

import analyze_cuda_benchmark as model


BENCHMARK = "minillm-cuda-profiler"
PROCESSES = (
    ("off-before", "none"),
    ("nsys", "nsys"),
    ("off-middle", "none"),
    ("ncu", "ncu"),
    ("off-after", "none"),
)
SELECTOR = dict(workload="chunked-prefill-1536", iteration=2, phase="forwards",
                call_in_phase=11, layer=27, kernel="pv_kernel")
NSYS_FLAGS = (
    "--trace=cuda", "--sample=none", "--cpuctxsw=none", "--discard-environment=true",
    "--cuda-memory-usage=true", "--force-overwrite=false", "--export=sqlite",
)
NCU_FLAGS = (
    "--target-processes", "application-only", "--kernel-name-base", "function",
    "--kernel-name", "regex:.*pv_kernel.*", "--launch-skip", "1567", "--launch-count", "1",
    "--set", "basic", "--replay-mode", "kernel", "--clock-control", "none",
    "--cache-control", "none", "--kill", "0",
)
RAW_NAMES = ("nsys.nsys-rep", "nsys.sqlite", "ncu.ncu-rep")
ENVIRONMENT_KEYS = {
    "PATH", "HOME", "USER", "TMPDIR", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG", "LANG", "LC_ALL",
}
MODEL_FILES = (
    "baseline-manifest.json", "baseline-source-state.json", "baseline-summary.json",
    "baseline-validation-summary.json", "baseline-reference.json",
)
METRICS = {
    "gpu__time_duration.sum": "ns",
    "sm__cycles_active.avg": "cycle",
    "dram__cycles_active.avg": "cycle",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "%",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "%",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "%",
    "profiler__replayer_passes": "pass",
}
require = model.require
read = model.read
sha = model.sha


def schedule():
    return [dict(name=name, profiler=profiler, order=index, report=name + ".json")
            for index, (name, profiler) in enumerate(PROCESSES)]


def flatten(report):
    result = []
    for work in report["workloads"]:
        for iteration in work["iterations"]:
            for phase in ("setup", "forwards"):
                for index, call in enumerate(iteration[phase]):
                    result.append(dict(workload=work["name"], iteration=iteration["index"], phase=phase,
                                       call_in_phase=index, measured=iteration["phase"] == "measured" and phase == "forwards",
                                       call=call))
    require(len(result) == 585, "Profiler 目标必须执行完整 585 次 forward")
    return result


def selected_call(report):
    calls = flatten(report)
    matches = [(index, value) for index, value in enumerate(calls)
               if all(value[key] == SELECTOR[key] for key in ("workload", "iteration", "phase", "call_in_phase"))]
    require(len(matches) == 1, "NCU 选择器没有唯一对应的模型调用")
    index, selected = matches[0]
    require(index * 28 + SELECTOR["layer"] == 1567, "NCU kernel 序号不符合固定 workload")
    require(selected["call"]["input_tokens"] == 128 and max(selected["call"]["context_after"]) == 1536,
            "NCU 目标不是 128-token、1536 context 的完整层")
    return index, selected


def product_sources(files):
    names = ("apps/cuda_runtime_bench.cpp", "apps/cuda_benchmark.h", "apps/cuda_reports.h", "apps/options.h")
    return {name: record["sha256"] for name, record in files.items()
            if name.startswith(("include/minillm/", "src/minillm/")) or name in names}


def verify_models(directory, manifest, files):
    baseline = read(model.artifact(directory, "baseline-manifest.json"))
    state = read(model.artifact(directory, "baseline-source-state.json"))
    summary = read(model.artifact(directory, "baseline-summary.json"))
    numeric = read(model.artifact(directory, "baseline-validation-summary.json"))
    reference = read(model.artifact(directory, "baseline-reference.json"))
    require(baseline["benchmark"] == model.BENCHMARK and baseline["reports"] == model.schedule(),
            "关联基线不是完整 70 进程协议")
    require(baseline["run_id"] == manifest["baseline"]["run_id"]
            and baseline["binary"]["sha256"] == manifest["binary"]["sha256"]
            and baseline["model"]["sha256"] == model.MODEL_SHA256
            and baseline["input"]["sha256"] == model.INPUT_SHA256, "Profiler 与基线的执行身份不同")
    require(sha(directory / "baseline-manifest.json") == manifest["baseline"]["manifest_sha256"]
            and sha(directory / "baseline-source-state.json") == baseline["source"]["worktree_state_sha256"],
            "关联基线的 manifest 或源码清单摘要不符")
    require(summary["run_id"] == baseline["run_id"] and summary["reports"] == 70
            and summary["data_path_gates"] == "passed"
            and summary["status"] in ("measured", "measurement_inconclusive"), "关联基线没有完整的正确性与数据路径结果")
    require(numeric["complete"] is True and numeric["passed"] is True and numeric["full_corpus_contract"] is True
            and numeric["totals"]["numeric_failures"] == 0, "关联完整数值门禁未通过")
    old = {item["path"]: item for item in state["files"]}
    require(product_sources(files) == product_sources(old), "模型执行源码不同，不能关联既有基线")
    spec = read(model.artifact(directory, "input.json"))
    expected = model.validate_report(reference, spec)
    require(reference["backend"] == "cuda" and reference["scope"] == "paired_model_baseline"
            and reference["run_identity"]["run_id"] == baseline["run_id"]
            and reference["run_identity"]["binary_sha256"] == manifest["binary"]["sha256"],
            "关联报告不是同一二进制的 CUDA 基线")
    selected_call(reference)
    return spec, expected


def kernel_role(name):
    if "minillm::cuda::" not in name:
        return "vendor_or_other"
    match = re.search(r"\b(reset|finite|gather|norm|rope|pointwise|argmax|store|qk|softmax|pv)_kernel\b", name)
    require(match is not None, "时间线包含未识别的项目 kernel：" + name)
    return match[1]


def required_kernels(call):
    selected = int(call["logits_rows"] > 0)
    return dict(reset=1, finite=28, gather=1 + selected, norm=112 + selected, rope=56, pointwise=84,
                argmax=selected, store=28, qk=28, softmax=28, pv=28)


def project_sequence(call):
    layer = ["norm", "norm", "norm", "rope", "rope", "store", "qk", "softmax", "pv",
             "pointwise", "norm", "pointwise", "pointwise", "finite"]
    return ["reset", "gather"] + layer * 28 + (["norm", "gather", "argmax"] if call["logits_rows"] else [])


def union_ns(intervals):
    result, end = 0, None
    for start, stop in sorted(intervals):
        require(model.integer(start) and model.integer(stop) and stop >= start, "设备时间区间无效")
        result += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return result


def analyze_nsys(path, report):
    calls = flatten(report)
    target_index, target = selected_call(report)
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        require(connection.execute("PRAGMA quick_check").fetchone()[0] == "ok", "NSys SQLite 完整性检查失败")
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        require({"StringIds", "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME",
                 "CUPTI_ACTIVITY_KIND_MEMCPY", "ENUM_CUDA_MEMCPY_OPER",
                 "TARGET_INFO_GPU", "META_DATA_CAPTURE"} <= tables,
                "NSys 没有完整 CUDA kernel、API、传输或设备信息")
        names = dict(connection.execute("SELECT id,value FROM StringIds"))
        devices = list(connection.execute("SELECT id,name,uuid,computeMajor,computeMinor FROM TARGET_INFO_GPU"))
        gpu = report["runtime"]["device"]
        require(any(row["id"] == 0 and row["name"] == gpu["name"]
                    and row["uuid"].lower().removeprefix("gpu-") == gpu["uuid"].lower()
                    and [row["computeMajor"], row["computeMinor"]] == gpu["compute_capability"] for row in devices),
                "NSys 设备身份与目标报告不同")
        diagnostics = [dict(row) for row in connection.execute("SELECT severity,text FROM DIAGNOSTIC_EVENT")] \
            if "DIAGNOSTIC_EVENT" in tables else []
        require(not any(re.search(r"lost|dropped|overflow|does not contain|not supported by this build", row["text"], re.I)
                        for row in diagnostics), "NSys 有丢失事件或不兼容诊断")
        grouped, events, roles, stream_ids = defaultdict(list), [], Counter(), set()
        windows = []
        current = None
        for row in connection.execute(
                "SELECT start,end,deviceId,contextId,streamId,correlationId,demangledName,gridX,gridY,gridZ,"
                "blockX,blockY,blockZ FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start,end"):
            event = dict(row)
            name = names[event.pop("demangledName")]
            role = kernel_role(name)
            require(model.integer(event["start"]) and model.integer(event["end"]) and event["end"] > event["start"],
                    "NSys kernel 时间无效")
            if role == "reset":
                if current is not None:
                    windows.append(current)
                current = dict(start=event["start"], correlation=event["correlationId"], events=[], transfers=[])
            if current is None:
                continue
            require(event["deviceId"] == 0, "NSys forward 使用了其他设备")
            event.update(name=name, role=role)
            current["events"].append(event)
            events.append(event)
            roles[role] += 1
            grouped[name].append(event["end"] - event["start"])
            if role != "vendor_or_other":
                stream_ids.add((event["deviceId"], event["contextId"], event["streamId"]))
        if current is not None:
            windows.append(current)
        require(len(windows) == len(calls) and len(stream_ids) == 1, "NSys 完整 forward 数或项目 stream 数不符")
        require(roles["vendor_or_other"] > 0, "NSys 缺少模型矩阵 kernel")
        transfers = list(connection.execute(
            "SELECT start,end,deviceId,streamId,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start,end"))
        window_index = 0
        for row in transfers:
            if row["start"] < windows[0]["start"]:
                continue
            while window_index + 1 < len(windows) and row["start"] >= windows[window_index + 1]["start"]:
                window_index += 1
            windows[window_index]["transfers"].append(dict(row))
        copy_kinds = dict(connection.execute("SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER"))
        selected_event = None
        results = []
        for index, (identity, window) in enumerate(zip(calls, windows)):
            call = identity["call"]
            counts = Counter(event["role"] for event in window["events"])
            require({name: counts[name] for name in required_kernels(call)} == required_kernels(call),
                    f"NSys 第 {index} 个 forward 缺少完整 28 层或输出 kernel")
            require([event["role"] for event in window["events"] if event["role"] != "vendor_or_other"]
                    == project_sequence(call), f"NSys 第 {index} 个 forward 的项目 kernel 顺序不符")
            h2d = d2h = 0
            for transfer in window["transfers"]:
                require(transfer["deviceId"] == 0 and model.integer(transfer["bytes"]), "NSys 传输字段无效")
                kind = copy_kinds[transfer["copyKind"]]
                if kind == "Host-to-Device":
                    h2d += transfer["bytes"]
                elif kind == "Device-to-Host":
                    d2h += transfer["bytes"]
                else:
                    require(False, "forward 中出现未归属的 CUDA 传输：" + kind)
            require(h2d == 12 * call["input_tokens"] + 4 * call["logits_rows"]
                    and d2h == 8 + 4 * call["logits_rows"], f"NSys 第 {index} 个 forward 的显式传输不符")
            intervals = [(value["start"], value["end"]) for value in window["events"] + window["transfers"]]
            span = max(end for _, end in intervals) - window["start"]
            busy = union_ns(intervals)
            require(0 < busy <= span, "NSys 设备活跃区间或 forward 跨度无效")
            values = {key: value for key, value in identity.items() if key != "call"}
            values.update(input_tokens=call["input_tokens"], logits_rows=call["logits_rows"],
                          context_before=call["context_before"], context_after=call["context_after"],
                          host_forward_to_token_ns=call["host_forward_to_token_ns"], device_span_ns=span,
                          device_busy_union_ns=busy, device_gap_ns=span - busy,
                          kernel_ns=sum(event["end"] - event["start"] for event in window["events"]),
                          kernel_count=len(window["events"]), h2d_bytes=h2d, d2h_bytes=d2h)
            results.append(values)
            if index == target_index:
                matching = [event for event in window["events"] if event["role"] == "pv"]
                selected_event = dict(matching[SELECTOR["layer"]])
        first_api = connection.execute(
            "SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=?",
            (windows[0]["correlation"],)).fetchone()[0]
        last_device = max(event["end"] for event in windows[-1]["events"] + windows[-1]["transfers"])
        require(model.integer(first_api), "NSys 缺少首个 forward 的 CUDA API 关联")
        api = []
        for row in connection.execute(
                "SELECT n.value AS name,COUNT(*) AS calls,SUM(a.end-a.start) AS elapsed_ns,"
                "SUM(CASE WHEN a.returnValue!=0 THEN 1 ELSE 0 END) AS nonzero_returns "
                "FROM CUPTI_ACTIVITY_KIND_RUNTIME a JOIN StringIds n ON n.id=a.nameId "
                "WHERE a.start>=? AND a.start<=? GROUP BY n.value ORDER BY elapsed_ns DESC", (first_api, last_device)):
            api.append(dict(row))
        allocation_apis = [row for row in api if re.match(r"cuda(Malloc|Free)", row["name"])]
        measured = [row for row in results if row["measured"]]
        return dict(schema_version=1, status="passed", complete_model=True, forwards=len(results),
                    measured_forwards=len(measured), layers_per_forward=28, project_streams=len(stream_ids),
                    kernel_count=len(events), kernel_roles=dict(roles),
                    kernels=[dict(name=name, calls=len(times), total_ns=sum(times), median_ns=statistics.median(times))
                             for name, times in sorted(grouped.items(), key=lambda item: -sum(item[1]))],
                    cuda_api=api, allocation_apis_in_forward_span=allocation_apis,
                    allocation_attribution="CUDA Runtime API 观察，不覆盖 Driver API；项目归属另由 Runtime 计数验证",
                    measured_device_span_ns=sum(row["device_span_ns"] for row in measured),
                    measured_device_busy_union_ns=sum(row["device_busy_union_ns"] for row in measured),
                    measured_device_gap_ns=sum(row["device_gap_ns"] for row in measured),
                    diagnostics=diagnostics, selected_kernel=dict(selector=SELECTOR, forward_index=target_index,
                    input_tokens=target["call"]["input_tokens"], context_after=target["call"]["context_after"],
                    event=selected_event), calls=results,
                    limitations=["外部 Profiler 时间线，不是无插桩模型基线",
                                 "设备空隙不单独证明 host 提交、驱动或调度是原因",
                                 "GPU 活跃区间不等同于硬件 SM 利用率或 DRAM 带宽"])
    finally:
        connection.close()


def analyze_ncu(path, nsys, report):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    require(len(rows) == 3 and len(rows[0]) == len(set(rows[0])) and all(len(row) == len(rows[0]) for row in rows),
            "NCU raw CSV 必须包含表头、单位和一个 kernel")
    units, values = dict(zip(rows[0], rows[1])), dict(zip(rows[0], rows[2]))
    require(values["ID"] == "0" and "minillm::cuda::" in values["Kernel Name"]
            and kernel_role(values["Kernel Name"]) == "pv" and values["Device"] == "0",
            "NCU 结果不是选定的项目 PV kernel")
    gpu = report["runtime"]["device"]
    require(values["CC"] == ".".join(str(value) for value in gpu["compute_capability"])
            and values["device__attribute_display_name"] == gpu["name"], "NCU 设备与目标报告不符")

    def number(name):
        value = float(values[name])
        require(model.finite(value) and value >= 0, "NCU 指标缺失或无效：" + name)
        return value

    metrics = {}
    for name, unit in METRICS.items():
        require(units[name] == unit, "NCU 指标单位不符：" + name)
        metrics[name] = dict(value=number(name), unit=unit)
    require(metrics["gpu__time_duration.sum"]["value"] > 0
            and metrics["profiler__replayer_passes"]["value"] >= 1, "NCU 没有实际耗时或回放信息")
    event = nsys["selected_kernel"]["event"]
    for name, column in (("grid", "grid"), ("block", "block")):
        for axis in ("x", "y", "z"):
            require(number(f"launch__{name}_dim_{axis}") == event[column + axis.upper()], "NCU 与 NSys 目标 launch 形状不同")
    return dict(schema_version=1, status="passed", kernel_name=values["Kernel Name"], selector=SELECTOR,
                profiled_kernel_count=1, metrics=metrics,
                replay_mode="kernel", cache_control="none", clock_control="none",
                launch=dict(grid=[event["gridX"], event["gridY"], event["gridZ"]],
                            block=[event["blockX"], event["blockY"], event["blockZ"]]),
                limitations=["NCU 指标来自单个 layer/kernel；不能外推全部形状",
                             "回放时间与缓存、频率状态不能替代无 Profiler 的模型性能"])


def overhead(reports, spec):
    audited = {name: model.validate_report(report, spec) for name, report in reports.items()}
    cases = {}
    for work in spec["workloads"]:
        name = work["name"]
        cases[name] = {}
        for profiler, before, after in (("nsys", "off-before", "off-middle"), ("ncu", "off-middle", "off-after")):
            values = {key: audited[key]["cases"][name]["median_ns"] for key in (profiler, before, after)}
            off = statistics.median((values[before], values[after]))
            cases[name][profiler] = dict(before_median_ns=values[before], after_median_ns=values[after],
                                        off_bracket_median_ns=off, on_median_ns=values[profiler],
                                        relative_percent=100 * (values[profiler] / off - 1),
                                        before_after_relative_percent=100 * (values[after] / values[before] - 1))
    return dict(schema_version=1, scope="相邻无 Profiler 进程夹住一个外部 Profiler 进程的诊断",
                formal_performance_baseline=False, independent_pairs=1, confidence_interval=None, cases=cases,
                limitations=["每种 Profiler 只有一个进程，不进行显著性或加速验收",
                             "每进程三次重复只用于取中位数，不视为独立 trial",
                             "NCU 选择的 measured repetition 受到回放影响，原始样本全部保留"])


def verify_command(process, slot, manifest):
    arguments = process["arguments"]
    require(process["order"] == slot["order"] and process["exit_code"] == 0 and process["profiler"] == slot["profiler"],
            "Profiler 进程身份或返回码不符")
    require(arguments[-8:] == ["--model", manifest["model"]["path"], "--input", manifest["input"]["execution_path"],
                               "--backend", "cuda", "--output", process["report_execution_path"]],
            "Profiler 的目标模型、冻结输入或后端命令不同")
    require(process["environment_policy"] == "allowlist_no_credentials", "Profiler 没有固定的环境白名单")
    environment = process["environment"]
    require(isinstance(environment, dict) and set(environment) <= ENVIRONMENT_KEYS
            and environment.get("LANG") == environment.get("LC_ALL") == "C"
            and all(isinstance(value, str) for value in environment.values()),
            "Profiler 环境不符合非凭据白名单")
    if slot["profiler"] == "nsys":
        require(len(arguments) == len(NSYS_FLAGS) + 11 and arguments[:1] == ["profile"]
                and arguments[1:1 + len(NSYS_FLAGS)] == list(NSYS_FLAGS)
                and arguments[-10].startswith("--output="),
                "NSys 采集选项或环境变量保护不同")
        require(process["executable"] == manifest["tools"]["nsys"]["path"]
                and arguments[-9] == manifest["binary"]["path"], "NSys 或其目标执行文件不同")
    elif slot["profiler"] == "ncu":
        require(len(arguments) == len(NCU_FLAGS) + 11 and arguments[:len(NCU_FLAGS)] == list(NCU_FLAGS)
                and arguments[-11] == "--export", "NCU 选择器、回放、缓存或频率策略不同")
        require(process["executable"] == manifest["tools"]["ncu"]["path"]
                and arguments[-9] == manifest["binary"]["path"], "NCU 或其目标执行文件不同")
    else:
        require(len(arguments) == 8 and process["executable"] == manifest["binary"]["path"],
                "无 Profiler 进程的执行文件不同")


def validate_bundle(directory):
    directory = directory.resolve()
    manifest = read(model.artifact(directory, "manifest.json"))
    require(model.identical(manifest["schema_version"], 1) and manifest["benchmark"] == BENCHMARK
            and model.identical(manifest["processes"], schedule())
            and model.identical(manifest["selector"], SELECTOR), "Profiler 协议或进程计划不符")
    require(manifest["environment"]["profiler_scope"] == "external_diagnostic"
            and manifest["input"]["sha256"] == model.INPUT_SHA256
            and manifest["model"]["sha256"] == model.MODEL_SHA256
            and model.digest(manifest["binary"]["sha256"]), "Profiler 身份或数值配置不符")
    require(manifest["tools"]["nsys"]["version"].startswith("NVIDIA Nsight Systems version 2026.1.3")
            and "2025.1.1" in manifest["tools"]["ncu"]["version"]
            and all(model.digest(manifest["tools"][name]["sha256"]) for name in ("nsys", "ncu")),
            "Profiler 工具版本或摘要不在已验证范围")
    files = model.source_archive(directory, manifest["source"])
    require(sha(model.artifact(directory, "verify.py")) == files["scripts/analyze_cuda_profiler.py"]["sha256"]
            and sha(model.artifact(directory, "analyze_cuda_benchmark.py")) == files["scripts/analyze_cuda_benchmark.py"]["sha256"],
            "Profiler 复核工具不属于归档源码")
    require(sha(model.artifact(directory, "input.json")) == model.INPUT_SHA256, "Profiler 固定输入摘要不符")
    artifacts = read(model.artifact(directory, "artifact-manifest.json"))
    require(artifacts["schema_version"] == 1 and artifacts["purpose"] == "archive_revalidation"
            and artifacts["manifest_sha256"] == sha(directory / "manifest.json")
            and artifacts["raw_availability"] == "bundled" and artifacts["raw_archive"] == "profiler-raw.zip"
            and artifacts["external_tools_required_for_archive_revalidation"] is False, "Profiler 产物索引身份不符")
    availability, names = [], set()
    for item in artifacts["artifacts"]:
        name = item["path"]
        require(name not in names and model.digest(item["sha256"]), "Profiler 产物重复或摘要无效")
        path = model.artifact(directory, name)
        require(sha(path) == item["sha256"] and path.stat().st_size == item["size_bytes"], "Profiler 产物内容不符：" + name)
        names.add(name)
        availability.append(dict(path=name, sha256=item["sha256"], size_bytes=item["size_bytes"], status="AVAILABLE"))
    mandatory = {"manifest.json", "source-state.json", "source-snapshot.zip", "verify.py", "analyze_cuda_benchmark.py",
                 "input.json", "collection-status.json", "ncu-metrics.csv", "ncu-export.process.json", "ncu-export.stderr.txt",
                 "profiler-raw.zip", *MODEL_FILES}
    for tool in ("nsys", "ncu"):
        mandatory.update(tool + "-version." + suffix for suffix in ("process.json", "stdout.txt", "stderr.txt"))
    for slot in schedule():
        mandatory.update(slot["report"] + suffix for suffix in ("", ".stdout.txt", ".stderr.txt", ".process.json"))
    require(mandatory <= names, "Profiler 产物索引缺少必需文件")
    for tool in ("nsys", "ncu"):
        process = read(directory / (tool + "-version.process.json"))
        output = (directory / (tool + "-version.stdout.txt")).read_text(encoding="utf-8").strip()
        require(process["exit_code"] == 0 and process["arguments"] == ["--version"]
                and process["executable"] == manifest["tools"][tool]["path"]
                and output == manifest["tools"][tool]["version"], "Profiler 工具版本原始输出不符")
    spec, expected = verify_models(directory, manifest, files)
    status = read(directory / "collection-status.json")
    require(status["status"] == "passed" and len(status["reports"]) == len(PROCESSES), "Profiler 采集不完整或有失败进程")
    reports, environments = {}, []
    for slot, collected in zip(schedule(), status["reports"]):
        require(collected["file"] == slot["report"] and collected["exit_code"] == 0
                and collected["sha256"] == sha(directory / slot["report"]), "Profiler 进程顺序或报告摘要不符")
        process = read(directory / (slot["report"] + ".process.json"))
        verify_command(process, slot, manifest)
        environments.append(process["environment"])
        report = read(directory / slot["report"])
        require(report["backend"] == "cuda" and report["scope"] == "diagnostic_single_process"
                and report["run_identity"] is None and report["process"] is None,
                "Profiler 报告不能混入正式配对性能数据")
        audited = model.validate_report(report, spec)
        require(all(audited["cases"][name]["token_ids"] == expected["cases"][name]["token_ids"] for name in expected["cases"]),
                "Profiler 开关或独立进程之间的输出不同")
        reports[slot["name"]] = report
    require(all(model.identical(environment, environments[0]) for environment in environments),
            "Profiler 开关进程的环境不同")
    export = read(directory / "ncu-export.process.json")
    require(export["exit_code"] == 0 and export["executable"] == manifest["tools"]["ncu"]["path"]
            and len(export["arguments"]) == 7 and export["arguments"][0] == "--import"
            and export["arguments"][-5:] == ["--csv", "--page", "raw", "--print-units", "base"],
            "NCU 原始指标导出命令或返回码不符")
    with tempfile.TemporaryDirectory(prefix="cuda-profiler-audit-") as temporary:
        database = Path(temporary) / "nsys.sqlite"
        extract_raw_database(directory / "profiler-raw.zip", artifacts["raw_members"], database)
        nsys = analyze_nsys(database, reports["nsys"])
    ncu = analyze_ncu(directory / "ncu-metrics.csv", nsys, reports["ncu"])
    return dict(nsys=nsys, ncu=ncu, overhead=overhead(reports, spec),
                availability=dict(schema_version=1, status="AVAILABLE", artifacts=availability))


def extract_raw_database(path, members, database):
    require({item["path"] for item in members} == set(RAW_NAMES) and len(members) == len(RAW_NAMES),
            "原始 Profiler 包缺件或重复")
    with zipfile.ZipFile(path) as archive:
        require(set(archive.namelist()) == set(RAW_NAMES) and len(archive.infolist()) == len(RAW_NAMES),
                "原始 Profiler ZIP 成员不符")
        for item in members:
            info = archive.getinfo(item["path"])
            require(model.integer(item["size_bytes"], 1) and model.digest(item["sha256"])
                    and not info.is_dir() and (info.external_attr >> 16) & 0o170000 != 0o120000
                    and info.file_size == item["size_bytes"] and info.file_size <= 4 * 1024**3,
                    "原始 Profiler 文件类型或尺寸无效")
        # 只将固定名称的数据库流式写入调用方的临时文件，不按 ZIP 路径展开。
        for item in members:
            digest = hashlib.sha256()
            with archive.open(item["path"]) as stream:
                if item["path"] == "nsys.sqlite":
                    with database.open("wb") as output:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                            output.write(block)
                else:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
            require(digest.hexdigest() == item["sha256"], "原始 Profiler 成员摘要不符：" + item["path"])


def analysis_text(result):
    nsys, ncu = result["nsys"], result["ncu"]
    lines = ["# CUDA 完整模型 Profiler", "",
             "- 独立外部诊断进程；原始无 Profiler 基线保持单独身份。",
             f"- NSys：{nsys['forwards']} 次完整 forward，每次 28 层，{nsys['kernel_count']} 次 kernel。",
             f"- 正式 workload 的测量部分在本时间线中共 {nsys['measured_forwards']} 次 forward。",
             "- device span、kernel time、GPU 活跃区间并集、host elapsed 与硬件计数器分别报告。",
             "- 每种 Profiler 只有一个进程；开关差异不构成显著性或加速结论。",
             "- 项目模型/KV/调度归项目，GEMM 归 cuBLAS；不包含 GPU Serving 或 PagedAttention。", "",
             "## Kernel 时间线", "", "| Kernel | 次数 | 总时间 ms | 中位 us |",
             "| --- | ---: | ---: | ---: |"]
    for kernel in nsys["kernels"]:
        lines.append(f"| `{kernel['name']}` | {kernel['calls']} | {kernel['total_ns']/1e6:.3f} | {kernel['median_ns']/1e3:.3f} |")
    lines += ["", "## NCU", "", f"- Kernel：`{ncu['kernel_name']}`。",
              "- 目标为 chunked-prefill-1536 第一次 measured repetition 的最后一个 chunk、最后一层 PV。"]
    for name, metric in ncu["metrics"].items():
        lines.append(f"- `{name}`：{metric['value']} {metric['unit']}。")
    lines += ["", "## Profiler 开关诊断", "", "| 用例 | NSys 相对变化 % | NCU 相对变化 % |",
              "| --- | ---: | ---: |"]
    for name, values in result["overhead"]["cases"].items():
        lines.append(f"| {name} | {values['nsys']['relative_percent']:.2f} | {values['ncu']['relative_percent']:.2f} |")
    lines += ["", "## 限制", "",
              "- 未锁频；进程边界遥测不是连续 GPU 频率或功耗曲线。",
              "- 选定 NCU kernel 的回放结果不能代替正常进程，也不能外推其他 shape。",
              "- 微基准的正序/逆序差异没有在这里重做受控实验，不能据模型时间线宣称已消除。",
              "- 关联基线摘要不替代完整模型与微基准归档；完整交付包须同时复核各归档。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--schedule", action="store_true")
    args = parser.parse_args()
    try:
        if args.schedule:
            print(json.dumps(dict(processes=schedule(), selector=SELECTOR, nsys_flags=NSYS_FLAGS, ncu_flags=NCU_FLAGS),
                             ensure_ascii=False, indent=2))
            return 0
        require(args.directory is not None, "必须指定 Profiler 证据目录")
        result = validate_bundle(args.directory)
        if args.write:
            model.publish(args.directory, {"nsys-summary.json": result["nsys"], "ncu-selected-kernel.json": result["ncu"],
                                          "profiler-overhead.json": result["overhead"], "analysis.md": analysis_text(result),
                                          "availability.json": result["availability"]})
        print(json.dumps(dict(status="passed", model_forwards=result["nsys"]["forwards"],
                              ncu_kernels=result["ncu"]["profiled_kernel_count"],
                              artifacts=len(result["availability"]["artifacts"])), ensure_ascii=False, indent=2))
        return 0
    except (model.ValidationError, KeyError, ValueError, TypeError, OSError, sqlite3.DatabaseError, zipfile.BadZipFile) as error:
        print(f"CUDA Profiler 证据复核失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
