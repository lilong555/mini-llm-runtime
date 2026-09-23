"""在线观测的完整性、时间守恒与 token 关联反例。"""

import copy
import importlib.util
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("analyzer", Path(__file__).parents[1] / "scripts/analyze_telemetry.py")
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


def fixture(mode="batches"):
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
    return rows, report, engine


passed = 0


def check(name, mutate=None, mode="batches"):
    global passed
    rows, report, engine = fixture(mode)
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
print(f"{passed}/{passed} tests passed")
