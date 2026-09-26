"""在线观测的完整性、时间守恒与 token 关联反例。"""

import copy
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

spec = importlib.util.spec_from_file_location("analyzer", Path(__file__).parents[1] / "scripts/analyze_telemetry.py")
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


def fixture(mode="batches", version=1, backend="minillm"):
    engine = dict(telemetry_mode=mode, telemetry_capacity=4, metrics_backend="minillm", max_active=2,
                  queue_capacity=4, context_tokens=16, block_size=1, max_model_len=8, batch_tokens=4)
    rows = [dict(type="header", schema_version=1, clock="engine_relative_steady_ns", mode=mode,
                 backend="minillm", policy="mixed", capacity=4, storage_bytes=1024, runtime_replay_available=False)]
    telemetry = []
    for index in range(2):
        start = 100 + index * 300
        item = dict(request_id="r", request_order=1, sequence=0, prefill=index == 0, tokens=1,
                    context_before=index, logits_tokens=1, token_index=index, sampled_token=42 + index,
                    emitted=True, emitted_ns=start + 150)
        runner = None
        if mode == "stages":
            stages = []
            names = ("kv_prepare embedding rope_prepare attention_norm query_projection key_projection "
                     "value_projection qk_norm_rope_kv attention output_projection attention_residual "
                     "ffn_norm gate_projection up_projection swiglu down_projection ffn_residual final_norm lm_head").split()
            for name in names:
                matrix = name.endswith("projection") or name == "lm_head"
                stages.append(dict(name=name, calls=1, wall_ns=1, matrix_m=int(matrix),
                                   matrix_n=int(matrix), matrix_k=int(matrix), varying_shape=False,
                                   parallel_wall_ns=0, caller_wait_ns=0, worker_work_sum_ns=0))
            runner = dict(completed=True, stages=stages, unaccounted_ns=1, forward_ns=20, sampling_ns=1)
        rows.append(dict(type="batch", batch_id=index + 1, completed=True, runner_completed=True,
                         start_ns=start, admission_ns=1, scheduler_ns=2, prepare_ns=3, runner_start_ns=start + 6,
                         runner_ns=100, finish_ns=start + 200, prefill_tokens=int(index == 0), decode_tokens=int(index == 1),
                         logits_tokens=1, sequences=1, waiting_requests=0, active_requests=1,
                         context_before_sum=index, context_before_max=index, context_after_sum=index + 1,
                         context_after_max=index + 1, reserved_unique_blocks=3,
                         resources_before=dict(live_kv_pages=index, resident_kv_payload_bytes=64),
                         resources_after=dict(live_kv_pages=index + 1, resident_kv_payload_bytes=64),
                         runner=runner, slices=[item]))
        telemetry.append(dict(batch_id=index + 1, request_order=1, token_index=index, engine_elapsed_ns=start + 150))
    rows.append(dict(type="footer", recorded=2, dropped=0, complete=True, engine_error="",
                     resources_final=dict(live_kv_pages=0, resident_kv_payload_bytes=64)))
    report = dict(server_before=dict(policy="mixed", batches=0), server_after=dict(batches=2),
                  requests=[dict(id="r", success=True, token_telemetry=telemetry, token_ids=[42, 43],
                                 token_times_ms=[1., 2.], usage=dict(prompt_tokens=1))])
    if version == 2:
        rows[0].update(schema_version=2, backend=backend,
                       capabilities=dict(max_sequences=2, max_batch_tokens=4, max_model_len=8,
                                         prefix_copy=backend != "minillm-cuda", runtime_stage_profile=backend == "minillm",
                                         synchronous_execute=True),
                       resource_boundaries=dict(before="before_execute", after="after_execute_before_request_cleanup",
                                                final="after_engine_stop_before_runner_destruction"))
        engine["metrics_backend"] = backend
        for row in rows[1:]:
            for key in ("resources_before", "resources_after", "resources_final"):
                if key not in row:
                    continue
                resource = row[key]
                if backend == "llama.cpp":
                    row[key] = None
                else:
                    cuda = backend == "minillm-cuda"
                    resource.update(layout="contiguous" if cuda else "paged", capacity_tokens=16,
                                    live_tokens=resource["live_kv_pages"] if cuda else None,
                                    owned_device_bytes=512 if cuda else None, state_valid=True, reusable=True)
                    if cuda:
                        resource["live_kv_pages"] = None
            if backend != "minillm" and "runner" in row:
                row["runner"] = None
    return rows, report, engine


passed = 0


def check(name, mutate=None, mode="batches", version=1, backend="minillm"):
    global passed
    rows, report, engine = fixture(mode, version, backend)
    if mutate:
        mutate(rows, report, engine)
    try:
        result = analyzer.validate_capture(rows, report, engine)
    except (ValueError, KeyError, TypeError, IndexError):
        if not mutate:
            raise
    else:
        if mutate:
            raise AssertionError(f"错误证据被接受：{name}")
        assert len(result[0]) == 2 and len(result[1]) == 1
    passed += 1
    print(f"[PASS] {name}")


check("batch-linked")
check("stage-linked", mode="stages")
check("missing-footer", lambda r, *_: r.pop())
check("missing-batch", lambda r, *_: r.pop(1))
check("duplicate-batch", lambda r, *_: r.insert(1, copy.deepcopy(r[1])))
check("buffer-overflow", lambda r, *_: r[-1].update(dropped=1))
check("incomplete-capture", lambda r, *_: r[-1].update(complete=False))
check("backend-failure", lambda r, *_: r[-1].update(engine_error="backend_error"))
check("incomplete-batch", lambda r, *_: r[1].update(completed=False))
check("boolean-integer", lambda r, *_: r[1].update(sequences=True))
check("invalid-stage-mode", lambda r, *_: r[0].update(mode="off"))
check("time-conservation", lambda r, *_: r[1].update(prepare_ns=99))
check("time-reversal", lambda r, *_: r[2].update(start_ns=1))
check("out-of-range-sequence", lambda r, *_: r[1]["slices"][0].update(sequence=2))
check("token-count", lambda r, *_: r[1].update(prefill_tokens=2))
check("kv-context", lambda r, *_: r[1].update(context_after_sum=2))
check("physical-pages", lambda r, *_: r[1].update(resources_before=None))
check("resource-leak", lambda r, *_: r[-1]["resources_final"].update(live_kv_pages=1))
check("no-client-token", lambda r, report, _: report["requests"][0]["token_telemetry"].pop())
check("wrong-client-token", lambda r, report, _: report["requests"][0]["token_ids"].__setitem__(1, 99))
check("wrong-client-time", lambda r, report, _: report["requests"][0]["token_telemetry"][1].update(engine_elapsed_ns=99))
check("reused-request-order", lambda r, *_: r[2]["slices"][0].update(request_id="other"))
check("missing-runtime", lambda r, *_: r[1].update(runner=None), mode="stages")
check("missing-stage", lambda r, *_: r[1]["runner"]["stages"].pop(), mode="stages")
check("runtime-time", lambda r, *_: r[1]["runner"].update(forward_ns=99), mode="stages")
check("matrix-shape", lambda r, *_: r[1]["runner"]["stages"][-1].update(matrix_m=2), mode="stages")
check("runtime-on-disabled", lambda r, *_: r[1].update(runner={}))

check("v2-cpu", version=2)
check("v2-upstream", mode="stages", version=2, backend="llama.cpp")
check("v2-cuda-no-fake-stages", mode="stages", version=2, backend="minillm-cuda")
for name, mutate in (
    ("cuda-no-fake-pages", lambda r, *_: r[1]["resources_after"].update(live_kv_pages=1)),
    ("cuda-live-capacity", lambda r, *_: r[1]["resources_after"].update(live_tokens=17)),
    ("cuda-physical-not-credits", lambda r, *_: r[1]["resources_before"].update(capacity_tokens=8)),
    ("cuda-resident-not-freed", lambda r, *_: r[-1]["resources_final"].update(resident_kv_payload_bytes=0)),
    ("cuda-poisoned-not-success", lambda r, *_: r[1]["resources_after"].update(state_valid=False, reusable=False, live_tokens=None)),
    ("cuda-capability", lambda r, *_: r[0]["capabilities"].update(prefix_copy=True)),
    ("cuda-commit-count", lambda r, *_: r[1]["resources_after"].update(live_tokens=2)),
):
    check(name, mutate, version=2, backend="minillm-cuda")

parser = argparse.ArgumentParser()
parser.add_argument("--server", type=Path)
parser.add_argument("--cuda-enabled", action="store_true")
args = parser.parse_args()
if args.server:
    for options, message in (
        (["--max-active", "8"], "4 sequences"),
        (["--batch-tokens", "256"], "128 batch tokens"),
        (["--max-model-len", "2049"], "2048 model length"),
        (["--context", "8208"], "context credits"),
        (["--prefix-entries", "4"], "prefix"),
        (["--prefix-tokens", "16"], "prefix"),
        (["--gpu-layers", "1"], "gpu_layers"),
        (["--kernel", "scalar"], "scalar"),
        ([], "cannot open GGUF file" if args.cuda_enabled else "MINILLM_ENABLE_CUDA=ON"),
    ):
        result = subprocess.run([str(args.server), "--backend", "mini-cuda", "--model",
                                 str(Path(__file__).with_name("absent-cuda-model.gguf")), *options],
                                capture_output=True, text=True, encoding="utf-8", timeout=10, check=False)
        assert result.returncode != 0 and message in result.stderr, (options, result.stderr)
        passed += 1
        print(f"[PASS] cuda-cli-{options or 'defaults'}")
print(f"{passed}/{passed} tests passed")
