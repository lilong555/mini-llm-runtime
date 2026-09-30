"""完整模型 Profiler 的确定性协议与反例检查；不执行 GPU 或 NVIDIA 采集工具。"""

from contextlib import redirect_stderr
from copy import deepcopy
import csv
import hashlib
import io
from pathlib import Path
import sqlite3
import sys
import tempfile
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_cuda_profiler as audit
import cuda_benchmark_validation_tests as model_fixture

PV_MANGLED = ("_ZN7minillm4cuda45_GLOBAL__N__404290e7_12_attention_cu_166305bf9pv_kernel"
              "ENS0_16DeviceTensorViewIKtEENS0_7KvShapeEmmNS2_IKiEES7_mNS2_IfEES8_Pi")


def invalid(function):
    try:
        function()
    except (audit.model.ValidationError, KeyError, ValueError, sqlite3.DatabaseError, zipfile.BadZipFile):
        return
    raise AssertionError("无效 Profiler 证据未被拒绝")


def protocol_and_selection_are_frozen():
    plan = audit.schedule()
    assert [row["profiler"] for row in plan] == ["none", "nsys", "none", "ncu", "none"]
    assert len({row["report"] for row in plan}) == 5
    assert "--discard-environment=true" in audit.NSYS_FLAGS
    assert audit.NCU_EXPORT_FLAGS[-4:] == ("--print-kernel-base", "mangled", "--rename-kernels", "0")
    report = model_fixture.cuda_fixture()
    index, selected = audit.selected_call(report)
    assert index == 55 and selected["call"]["context_after"] == [1536, 0, 0, 0]
    assert len(audit.flatten(report)) == 585 and index * 28 + audit.SELECTOR["layer"] == 1567
    report["workloads"].pop()
    invalid(lambda: audit.flatten(report))


def gpu():
    return dict(name="fixture GPU", uuid="12345678-1234-1234-1234-123456789abc",
                compute_capability=[8, 9])


def calls():
    return [
        dict(workload="prefill-16", iteration=0, phase="forwards", call_in_phase=0, measured=False,
             call=dict(logits_rows=1, input_tokens=16, context_before=[0, 0, 0, 0],
                       context_after=[16, 0, 0, 0], host_forward_to_token_ns=100000)),
        dict(workload="chunked-prefill-1536", iteration=2, phase="forwards", call_in_phase=11, measured=True,
             call=dict(logits_rows=1, input_tokens=128, context_before=[1408, 0, 0, 0],
                       context_after=[1536, 0, 0, 0], host_forward_to_token_ns=100000)),
    ]


def database(path):
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY,value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
            start INTEGER,end INTEGER,deviceId INTEGER,contextId INTEGER,streamId INTEGER,
            correlationId INTEGER,demangledName INTEGER,mangledName INTEGER,gridX INTEGER,gridY INTEGER,gridZ INTEGER,
            blockX INTEGER,blockY INTEGER,blockZ INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
            start INTEGER,end INTEGER,correlationId INTEGER,nameId INTEGER,returnValue INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
            start INTEGER,end INTEGER,deviceId INTEGER,streamId INTEGER,bytes INTEGER,copyKind INTEGER);
        CREATE TABLE TARGET_INFO_GPU(id INTEGER,name TEXT,uuid TEXT,computeMajor INTEGER,computeMinor INTEGER);
        CREATE TABLE META_DATA_CAPTURE(name TEXT,value TEXT);
        CREATE TABLE DIAGNOSTIC_EVENT(severity INTEGER,text TEXT);
        CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER,label TEXT);
    """)
    connection.execute("INSERT INTO TARGET_INFO_GPU VALUES(0,?,?,8,9)", (gpu()["name"], gpu()["uuid"]))
    connection.executemany("INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES(?,?)",
                           ((1, "Host-to-Device"), (2, "Device-to-Host")))
    name_ids = {}

    def name_id(name):
        if name not in name_ids:
            name_ids[name] = len(name_ids) + 1
            connection.execute("INSERT INTO StringIds VALUES(?,?)", (name_ids[name], name))
        return name_ids[name]

    clock = 100
    for item in calls():
        call = item["call"]
        for role in audit.project_sequence(call):
            name = f"void minillm::cuda::detail::{role}_kernel()" if role != "pointwise" else \
                "void minillm::cuda::detail::pointwise_kernel<true>()"
            symbol = PV_MANGLED if role == "pv" else "fixture_symbol_" + role
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                               (clock, clock + 10, 0, 1, 7, clock, name_id(name), name_id(symbol),
                                call["input_tokens"] * 16, 1, 1, 256, 1, 1))
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,0)",
                               (clock - 1, clock, clock, name_id("cudaLaunchKernel")))
            clock += 20
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (clock, clock + 10, 0, 1, 7, clock, name_id("ampere_sgemm_fixture"),
                            name_id("ampere_sgemm_fixture"), 1, 1, 1, 256, 1, 1))
        clock += 20
        for kind, size in ((1, 12 * call["input_tokens"] + 4 * call["logits_rows"]),
                           (2, 8 + 4 * call["logits_rows"])):
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,0,7,?,?)",
                               (clock, clock + 5, size, kind))
            clock += 20
    connection.commit()
    connection.close()


def nsys_result(path):
    items = calls()
    with patch.object(audit, "flatten", return_value=items), \
         patch.object(audit, "selected_call", return_value=(1, items[1])):
        return audit.analyze_nsys(path, dict(runtime=dict(device=gpu())))


def timeline_checks_complete_layers_and_transfers():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.sqlite"
        database(path)
        result = nsys_result(path)
        assert result["forwards"] == 2 and result["measured_forwards"] == 1
        assert result["kernel_roles"]["pv"] == 56 and result["project_streams"] == 1
        assert result["measured_device_gap_ns"] > 0
        assert result["selected_kernel"]["event"]["gridX"] == 2048
        assert result["calls"][1]["h2d_bytes"] == 1540 and result["calls"][1]["d2h_bytes"] == 12
        assert result["allocation_apis_in_forward_span"] == []


def serving_batch_mapping_and_no_runtime_rebuild():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.sqlite"
        database(path)
        batches = [dict(batch_id=i + 1, prefill_tokens=item["call"]["input_tokens"], decode_tokens=0,
                        logits_tokens=1, runner_ns=item["call"]["host_forward_to_token_ns"],
                        slices=[dict(context_before=item["call"]["context_before"][0], tokens=item["call"]["input_tokens"])])
                   for i, item in enumerate(calls())]
        report = dict(server_before=dict(batches=1))
        result = audit.analyze_nsys(path, report, batches, gpu())
        assert result["forwards"] == 2 and result["selected_kernel"] is None
        assert result["calls"][1]["batch_id"] == 2 and result["calls"][1]["measured"]
        wrong = deepcopy(batches)
        wrong[1]["logits_tokens"] = 2
        invalid(lambda: audit.analyze_nsys(path, report, wrong, gpu()))
        connection = sqlite3.connect(path)
        connection.execute("INSERT INTO StringIds VALUES(999,'cudaMalloc')")
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(120,130,999,999,0)")
        connection.commit()
        connection.close()
        invalid(lambda: audit.analyze_nsys(path, report, batches, gpu()))


def incomplete_kernel_trace_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.sqlite"
        database(path)
        connection = sqlite3.connect(path)
        connection.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE rowid=(SELECT rowid FROM "
                           "CUPTI_ACTIVITY_KIND_KERNEL WHERE demangledName=(SELECT id FROM StringIds "
                           "WHERE value LIKE '%finite_kernel%') LIMIT 1)")
        connection.commit()
        connection.close()
        invalid(lambda: nsys_result(path))


def wrong_stream_or_device_is_rejected():
    for statement in ("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET streamId=9 WHERE rowid=2",
                      "UPDATE TARGET_INFO_GPU SET uuid='wrong'",
                      "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET deviceId=1 WHERE rowid=2"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            database(path)
            connection = sqlite3.connect(path)
            connection.execute(statement)
            connection.commit()
            connection.close()
            invalid(lambda: nsys_result(path))


def wrong_layer_order_is_rejected_even_with_matching_counts():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.sqlite"
        database(path)
        connection = sqlite3.connect(path)
        first = connection.execute("SELECT demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE rowid=2").fetchone()[0]
        second = connection.execute("SELECT demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE rowid=3").fetchone()[0]
        connection.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET demangledName=? WHERE rowid=2", (second,))
        connection.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET demangledName=? WHERE rowid=3", (first,))
        connection.commit()
        connection.close()
        invalid(lambda: nsys_result(path))


def copy_errors_and_dropped_events_are_rejected():
    for statement in ("UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET bytes=bytes+4 WHERE rowid=1",
                      "INSERT INTO DIAGNOSTIC_EVENT VALUES(2,'CUDA kernel events dropped')"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            database(path)
            connection = sqlite3.connect(path)
            connection.execute(statement)
            connection.commit()
            connection.close()
            invalid(lambda: nsys_result(path))


def timeline_union_does_not_double_count_overlap():
    assert audit.union_ns([(1, 4), (2, 6), (8, 10)]) == 7
    assert audit.union_ns([(1, 4), (1, 4)]) == 3
    invalid(lambda: audit.union_ns([(2, 1)]))


def metrics(path):
    names = ["ID", "Kernel Name", "Device", "CC", "device__attribute_display_name", *audit.METRICS]
    units = [""] * 5 + list(audit.METRICS.values())
    values = ["0", PV_MANGLED, "0", "8.9", gpu()["name"]] + ["10"] * len(audit.METRICS)
    for kind, dims in (("grid", (2048, 1, 1)), ("block", (256, 1, 1))):
        for axis, size in zip(("x", "y", "z"), dims):
            names.append(f"launch__{kind}_dim_{axis}")
            units.append("")
            values.append(str(size))
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows((names, units, values))
    return names, units, values


def ncu_result(path):
    event = dict(gridX=2048, gridY=1, gridZ=1, blockX=256, blockY=1, blockZ=1,
                 name="minillm::cuda::<unnamed>::pv_kernel()", mangled_name=PV_MANGLED)
    return audit.analyze_ncu(path, dict(selected_kernel=dict(event=event)), dict(runtime=dict(device=gpu())))


def ncu_checks_one_kernel_and_hardware_units():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "metrics.csv"
        metrics(path)
        result = ncu_result(path)
        assert result["profiled_kernel_count"] == 1 and result["launch"]["grid"] == [2048, 1, 1]
        assert result["metrics"]["gpu__time_duration.sum"] == dict(value=10, unit="ns")
        assert result["clock_control"] == "none" and result["cache_control"] == "none"


def ncu_bad_units_counts_values_and_shapes_are_rejected():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "metrics.csv"
        for change in ("units", "nonfinite", "shape", "count", "name", "simplified_name", "duplicate"):
            names, units, values = metrics(path)
            if change == "units":
                units[names.index("gpu__time_duration.sum")] = "us"
            elif change == "nonfinite":
                values[names.index("gpu__time_duration.sum")] = "nan"
            elif change == "shape":
                values[names.index("launch__grid_dim_x")] = "16"
            elif change == "name":
                values[names.index("Kernel Name")] = "cuda_smoke"
            elif change == "simplified_name":
                values[names.index("Kernel Name")] = "unnamed>::pv_kernel()"
            elif change == "duplicate":
                names[-1] = names[0]
            rows = [names, units, values] + ([values] if change == "count" else [])
            with path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerows(rows)
            invalid(lambda: ncu_result(path))


def raw_bundle_is_closed_and_stream_verifiable():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive = root / "raw.zip"
        members = []
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
            for name in audit.RAW_NAMES:
                value = ("fixture " + name).encode()
                output.writestr(name, value)
                members.append(dict(path=name, sha256=hashlib.sha256(value).hexdigest(), size_bytes=len(value)))
        target = root / "database"
        audit.extract_raw_database(archive, members, target)
        assert target.read_bytes() == b"fixture nsys.sqlite"
        invalid(lambda: audit.extract_raw_database(archive, members[:-1], target))
        broken = deepcopy(members)
        broken[0]["sha256"] = "0" * 64
        invalid(lambda: audit.extract_raw_database(archive, broken, target))
        broken = deepcopy(members)
        broken[0]["path"] = "../escape"
        invalid(lambda: audit.extract_raw_database(archive, broken, target))


def raw_bundle_rejects_symlinks_and_extra_members():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for mode in ("symlink", "extra"):
            members = []
            archive = root / (mode + ".zip")
            with zipfile.ZipFile(archive, "w") as output:
                for name in audit.RAW_NAMES:
                    value = b"data"
                    entry = zipfile.ZipInfo(name)
                    if mode == "symlink" and name == "nsys.sqlite":
                        entry.external_attr = 0o120777 << 16
                    output.writestr(entry, value)
                    members.append(dict(path=name, sha256=hashlib.sha256(value).hexdigest(), size_bytes=4))
                if mode == "extra":
                    output.writestr("../escape", b"data")
            invalid(lambda: audit.extract_raw_database(archive, members, root / "database"))


def diagnostic_overhead_has_no_significance_claim():
    spec = dict(workloads=[dict(name="prefill")])
    reports = {name: dict(cases=dict(prefill=dict(median_ns=latency)))
               for name, latency in (("off-before", 100), ("nsys", 240), ("off-middle", 140),
                                     ("ncu", 450), ("off-after", 160))}
    with patch.object(audit.model, "validate_report", side_effect=lambda report, _: report):
        result = audit.overhead(reports, spec)
    assert result["confidence_interval"] is None and result["formal_performance_baseline"] is False
    assert result["cases"]["prefill"]["nsys"]["off_bracket_median_ns"] == 120
    assert result["cases"]["prefill"]["nsys"]["relative_percent"] == 100
    assert result["cases"]["prefill"]["ncu"]["relative_percent"] == 200


def external_profiler_commands_are_explicit():
    manifest = dict(binary=dict(path="/bin/target"), model=dict(path="/model"),
                    input=dict(execution_path="/input"),
                    tools=dict(nsys=dict(path="/bin/nsys"), ncu=dict(path="/bin/ncu")))
    target = ["--model", "/model", "--input", "/input", "--backend", "cuda", "--output", "/report"]
    for slot in audit.schedule():
        profiler = slot["profiler"]
        prefix = ["profile", *audit.NSYS_FLAGS, "--output=/raw/nsys", "/bin/target"] if profiler == "nsys" else \
            [*audit.NCU_FLAGS, "--export", "/raw/ncu", "/bin/target"] if profiler == "ncu" else []
        process = dict(order=slot["order"], exit_code=0, profiler=profiler,
                       environment_policy="allowlist_no_credentials", report_execution_path="/report",
                       environment=dict(LANG="C", LC_ALL="C", PATH="/bin"),
                       executable="/bin/" + profiler if profiler != "none" else "/bin/target",
                       arguments=prefix + target)
        audit.verify_command(process, slot, manifest)
        broken = deepcopy(process)
        broken["environment_policy"] = "inherited_all"
        invalid(lambda: audit.verify_command(broken, slot, manifest))
        broken = deepcopy(process)
        broken["environment"]["UNREGISTERED_VARIABLE"] = "fixture"
        invalid(lambda: audit.verify_command(broken, slot, manifest))
        broken = deepcopy(process)
        broken["arguments"][-5] = "/different-input"
        invalid(lambda: audit.verify_command(broken, slot, manifest))
        if profiler == "nsys":
            broken = deepcopy(process)
            broken["arguments"].remove("--discard-environment=true")
            invalid(lambda: audit.verify_command(broken, slot, manifest))
        if profiler != "none":
            broken = deepcopy(process)
            broken["arguments"].insert(-9, "--unregistered-option")
            invalid(lambda: audit.verify_command(broken, slot, manifest))


def failed_revalidation_preserves_previous_summary():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        original = b'{"status":"old"}\n'
        (root / "nsys-summary.json").write_bytes(original)
        with patch.object(sys, "argv", ["verify.py", "--directory", directory, "--write"]), redirect_stderr(io.StringIO()):
            assert audit.main() == 1
        assert (root / "nsys-summary.json").read_bytes() == original
        assert sorted(path.name for path in root.iterdir()) == ["nsys-summary.json"]


if __name__ == "__main__":
    tests = [protocol_and_selection_are_frozen, timeline_checks_complete_layers_and_transfers,
             serving_batch_mapping_and_no_runtime_rebuild,
             incomplete_kernel_trace_is_rejected, wrong_stream_or_device_is_rejected,
             wrong_layer_order_is_rejected_even_with_matching_counts,
             copy_errors_and_dropped_events_are_rejected, timeline_union_does_not_double_count_overlap,
             ncu_checks_one_kernel_and_hardware_units, ncu_bad_units_counts_values_and_shapes_are_rejected,
             raw_bundle_is_closed_and_stream_verifiable, raw_bundle_rejects_symlinks_and_extra_members,
             diagnostic_overhead_has_no_significance_claim, external_profiler_commands_are_explicit,
             failed_revalidation_preserves_previous_summary]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"{len(tests)}/{len(tests)} tests passed")
