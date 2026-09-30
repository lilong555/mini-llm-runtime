"""复核完整 CUDA 数值报告；只读原始文件，不覆盖已有验收结果。"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys


REFERENCES = ("cpu", "llama_f32", "cuda_canonical")
SCOPE = "CUDA-VS-001 Step 8 numerical"


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"JSON 字段重复：{key}")
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def artifact(directory, name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name,
            f"证据路径无效：{name}")
    relative = Path(name)
    require(not relative.is_absolute() and all(part not in ("", ".", "..") for part in name.split("/")),
            f"证据路径越界：{name}")
    path = Path(directory)
    for part in relative.parts:
        path = path / part
        require(not path.is_symlink(), f"证据不能依赖符号链接：{name}")
    require(path.is_file(), f"缺少证据：{name}")
    return path


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def check_score(score, vocabulary):
    require(integer(score["token"]) and score["token"] < vocabulary, "输出 token 无效")
    require(finite(score["margin"]) and score["margin"] >= 0, "argmax margin 无效")
    require(digest(score["sha256"]), "logits 摘要无效")


def compare_record(record, thresholds, expected=None, actual=None):
    require(record["all_finite"] is True, "logits 含非有限数")
    for name in ("rmse", "max_absolute", "cosine", "reference_margin"):
        require(finite(record[name]), f"数值指标无效：{name}")
    require(0 <= record["rmse"] <= record["max_absolute"] + 1e-12, "RMSE 与最大误差不一致")
    require(-1 <= record["cosine"] <= 1 + 1e-12 and record["reference_margin"] >= 0, "cosine 或 margin 越界")
    require(digest(record["actual_sha256"]) and digest(record["reference_sha256"]), "比较摘要无效")
    require(integer(record["actual_argmax"]) and integer(record["reference_argmax"]), "argmax 无效")
    near = record["reference_margin"] <= 2 * record["max_absolute"]
    equal = record["actual_argmax"] == record["reference_argmax"]
    passed = (record["rmse"] < thresholds["rmse_exclusive"]
              and record["max_absolute"] < thresholds["max_absolute_exclusive"]
              and record["cosine"] >= thresholds["cosine_min_inclusive"] and (near or equal))
    require(record["near_tie"] is near and record["argmax_equal"] is equal, "near-tie 或 argmax 声明不符")
    require(record["passed"] is passed, "数值通过状态与冻结门槛不符")
    for score, prefix in ((expected, "reference"), (actual, "actual")):
        if score is not None:
            require(record[f"{prefix}_sha256"] == score["sha256"]
                    and record[f"{prefix}_argmax"] == score["token"], "比较与对应 logits 摘要不符")
    if expected is not None:
        require(record["reference_margin"] == expected["margin"], "参照 margin 不符")
    return passed


def reference_settings(report, summary):
    expected = dict(flash_attention=False, kv_dtype="F16", threads=8,
                    qk_accumulation_dtype="F32", pv_accumulation_dtype="F32")
    require(report["references"]["llama_f32_settings"] == expected
            and summary["reference_attention"] == "unfused", "F32 参照必须显式使用 FP32 累加的非融合 attention")


def expanded(contract, corpus, length):
    seed = contract["corpus"][corpus]["seed_token_ids"]
    return [seed[p % len(seed)] for p in range(length)]


def batches(contract, corpus, length, chunk, sequences):
    positions = set(contract["teacher_forcing"]["positions"])
    rows = []
    for position in range(length):
        for sequence in range(sequences):
            seed = contract["corpus"][(corpus + sequence) % len(contract["corpus"])]["seed_token_ids"]
            rows.append((seed[position % len(seed)], position, sequence, int(position in positions)))
    return [rows[first:first + chunk] for first in range(0, len(rows), chunk)]


def input_digest(groups):
    state = hashlib.sha256()
    for batch in groups:
        state.update(struct.pack("<i", len(batch)))
        for row in batch:
            state.update(struct.pack("<4i", *row))
    return state.hexdigest()


def case_plan(contract):
    result = []
    recipe = contract["teacher_forcing"]
    for sequences in recipe["sequence_counts"]:
        for corpus, item in enumerate(contract["corpus"]):
            for length in recipe["lengths"]:
                for chunk in recipe["chunk_tokens"]:
                    groups = batches(contract, corpus, length, chunk, sequences)
                    result.append({
                        "id": f"{item['id']}-l{length}-c{chunk}-s{sequences}", "corpus": corpus,
                        "length": length, "chunk": chunk, "sequences": sequences, "batches": len(groups),
                        "sampled_rows": sum(row[3] for batch in groups for row in batch),
                        "input_sha256": input_digest(groups),
                    })
    return result


class Metrics:
    def __init__(self):
        self.totals = dict(teacher_cases=0, teacher_rows=0, teacher_comparisons=0,
                           generation_comparisons=0, numeric_failures=0, argmax_divergences=0, near_ties=0)
        self.extrema = dict(rmse_max=0.0, absolute_max=0.0, cosine_min=1.0)
        self.first_failure = None
        self.first_divergence = None
        self.by_reference = {name: dict(comparisons=0, numeric_failures=0, argmax_divergences=0,
                                        near_ties=0, rmse_max=0.0, absolute_max=0.0, cosine_min=1.0)
                             for name in REFERENCES}

    def add(self, record, teacher):
        self.totals["teacher_comparisons" if teacher else "generation_comparisons"] += 1
        group = self.by_reference[record["reference"]]
        group["comparisons"] += 1
        for key, condition in (("numeric_failures", not record["passed"]),
                               ("argmax_divergences", not record["argmax_equal"]),
                               ("near_ties", record["near_tie"])):
            self.totals[key] += condition
            group[key] += condition
        for target in (self.extrema, group):
            target["rmse_max"] = max(target["rmse_max"], record["rmse"])
            target["absolute_max"] = max(target["absolute_max"], record["max_absolute"])
            target["cosine_min"] = min(target["cosine_min"], record["cosine"])
        if not record["passed"] and self.first_failure is None:
            self.first_failure = record
        if not record["argmax_equal"] and self.first_divergence is None:
            self.first_divergence = record


class Transfers:
    def __init__(self):
        self.inputs = self.rows = self.debug_rows = self.forwards = 0

    def add(self, inputs, rows, forwards, debug=True):
        self.inputs += inputs
        self.rows += rows
        self.debug_rows += rows if debug else 0
        self.forwards += forwards

    def check(self, value, baseline, lengths, vocabulary):
        sequences = len(lengths)
        require(value["state"] == "ready" and value["post_launch_failures"] == 0, "Runtime 非 ready 或发生设备失败")
        require(value["sequence_lengths"] == lengths and value["live_kv_tokens"] == sum(lengths)
                and value["live_sequences"] == sum(length > 0 for length in lengths)
                and value["kv_capacity_tokens"] == sequences * 2048, "KV 状态不符")
        for name in ("resident", "weight_h2d_bytes", "rope_h2d_bytes", "owned_device_allocations",
                     "owned_device_bytes", "model_load_ns", "storage_initialization_ns", "weight_decode_upload_ns"):
            require(value[name] == baseline[name], f"稳态初始化字段发生变化：{name}")
        require(value["metadata_h2d_bytes"] == 12 * self.inputs + 4 * self.rows, "metadata 上传计数不符")
        require(value["token_d2h_bytes"] == 4 * self.rows and value["status_d2h_bytes"] == 8 * self.forwards,
                "token/status 下载计数不符")
        require(value["debug_d2h_bytes"] == 4 * vocabulary * self.debug_rows, "debug logits 下载计数不符")
        require(value["completed_forwards"] == self.forwards, "完成 forward 数量不符")
        require(value["intermediate_h2d_bytes"] == value["intermediate_d2h_bytes"] == 0, "hidden 存在 host 传输")


def generation_rows(report, references, contract, metrics):
    length, identity = report["prompt_tokens"], report["id"]
    require(len(report["rows"]) == 32, "生成步数不符")
    common = dict(cpu=True, llama_f32=True)
    first = dict(cpu=None, llama_f32=None)
    passed = True
    for step, row in enumerate(report["rows"]):
        require(row["step"] == step and set(row["references"]) == set(common), "生成行身份不符")
        check_score(row["cuda"], contract["model"]["vocabulary"])
        for backend in common:
            entry = row["references"][backend]
            require(entry["score"] == references[backend][step], "生成参照行不符")
            check_score(entry["score"], contract["model"]["vocabulary"])
            require(entry["input_prefix_equal"] is common[backend], "生成输入历史状态不符")
            if not common[backend]:
                require(entry["comparison"] is None, "输入历史已分歧，不能继续作 logits 数值比较")
                continue
            comparison = entry["comparison"]
            require(isinstance(comparison, dict), "缺少共同输入的 logits 比较")
            require(all(comparison[key] == value for key, value in dict(
                case=identity, reference=backend, step=step, sequence=0, position=length - 1 + step).items()),
                "生成比较身份不符")
            current_passed = compare_record(comparison, contract["thresholds"], entry["score"], row["cuda"])
            passed = passed and current_passed
            metrics.add(comparison, False)
            if row["cuda"]["token"] != entry["score"]["token"]:
                first[backend] = comparison
                common[backend] = False
    require(report["first_divergence"] == first and report["passed"] is passed, "生成首处分歧或状态不符")
    return first, passed


class Audit:
    def __init__(self, directory, expected_contract):
        self.directory = Path(directory)
        self.contract = read(artifact(directory, "validation-contract.json"))
        require(self.contract == expected_contract, "数值契约与冻结版本不符")
        self.report = self.load("full-validation.json")
        self.summary = self.load("validation-summary.json")
        self.input = self.load("input.json")
        self.plan = case_plan(self.contract)
        self.metrics = Metrics()
        self.vocabulary = self.contract["model"]["vocabulary"]
        self.canonical = {}
        self.transfers = {s: Transfers() for s in (1, 2, 4)}
        self.configurations = {}

    def load(self, name):
        return read(artifact(self.directory, name))

    def inputs(self):
        require(self.input["schema_version"] == 1 and self.input["contract_id"] == self.contract["contract_id"],
                "输入契约身份不符")
        require(self.input["teacher_cases"] == self.plan and len(self.plan) == 240, "teacher-forcing 输入集合不符")
        corpora = [dict(corpus=c, length=length, token_ids=expanded(self.contract, c, length))
                   for c in range(len(self.contract["corpus"])) for length in self.contract["teacher_forcing"]["lengths"]]
        require(self.input["corpora"] == corpora, "展开的 token 语料不符")
        require(self.input["canonical"] == dict(sequences=1, chunk=128), "独立参照配置不符")
        require(self.input["chunk_unit"] == "total_batch_input_rows"
                and self.input["token_order"] == "position_then_sequence"
                and self.input["sequence_corpus"] == "(case_corpus_index + sequence) % corpus_count"
                and self.input["input_digest"] == "sha256_i32le_batch_size_then_token_position_sequence_logits",
                "输入批次或摘要语义不符")
        natural = self.input["natural_generation"]
        require(natural["prompt_lengths"] == [16, 128, 1536] and natural["output_tokens"] == 32
                and natural["chunk"] == 128 and natural["eog_policy"] == "固定生成 32 个 token，不因 EOG 提前结束",
                "自然生成协议不符")

    def configuration(self):
        require([c["max_sequences"] for c in self.report["configurations"]] == [1, 2, 4], "CUDA 配置集合不符")
        for config in self.report["configurations"]:
            s = config["max_sequences"]
            require(config["max_model_len"] == 2048 and config["batch_tokens"] == 128
                    and config["passed"] is True and config["steady_project_allocation_calls"] == 0
                    and config["steady_project_release_calls"] == 0, "CUDA 配置或稳态设备分配不符")
            before = config["before"]
            p = before["resident"]
            require(before["owned_device_bytes"] == p["total_owned_bytes"] and before["owned_device_allocations"] == 4,
                    "实际设备分配与计划不符")
            require(p["total_owned_bytes"] == p["weights_bytes"] + p["workspace_bytes"] + p["kv_bytes"]
                    + p["library_workspace_bytes"], "显存计划总量不符")
            require(p["workspace_bytes"] == sum(p[k] for k in (
                "activations_bytes", "attention_scratch_bytes", "logits_bytes", "metadata_bytes", "rope_bytes", "padding_bytes")),
                "workspace 分项不符")
            require(p["kv_bytes"] == 28 * 2 * s * 2048 * 8 * 128 * 2
                    and p["weights_bytes"] == before["weight_h2d_bytes"] > 0
                    and p["rope_bytes"] == before["rope_h2d_bytes"] > 0, "KV 容量或初始化上传不符")
            require(all(value == self.contract["arithmetic"][key] for key, value in config["arithmetic"].items())
                    and set(config["arithmetic"]) == set(self.contract["arithmetic"]) - {"source_tensor_dtypes"},
                    "算术模式不符")
            device = config["device"]
            require(device["compute_capability"] == [8, 9] and device["name"]
                    and re.fullmatch(r"[0-9a-f-]{36}", device["uuid"]) is not None
                    and all(integer(device[k], 1) for k in ("driver_version", "runtime_version", "cublas_version")),
                    "设备或版本身份缺失")
            Transfers().check(before, before, [0] * s, self.vocabulary)
            self.configurations[s] = config
        require(self.load("memory-plan.json") == self.configurations[4]["before"]["resident"], "S=4 显存计划不符")

    def references(self):
        expected = []
        for backend in REFERENCES:
            for corpus in range(len(self.contract["corpus"])):
                for length in self.contract["teacher_forcing"]["lengths"]:
                    identity = f"{self.contract['corpus'][corpus]['id']}-l{length}-c128-s1"
                    file = f"canonical/{backend}-{identity}.json"
                    expected.append(dict(backend=backend, id=identity, file=file))
                    data = self.load(file)
                    groups = batches(self.contract, corpus, length, 128, 1)
                    require(data["schema_version"] == 1 and data["backend"] == backend and data["id"] == identity
                            and data["corpus"] == corpus and data["length"] == length
                            and data["input_sha256"] == input_digest(groups), "独立参照身份或输入不符")
                    positions = [p for p in self.contract["teacher_forcing"]["positions"] if p < length]
                    require([r["position"] for r in data["samples"]] == positions, "独立参照采样位置不符")
                    for row in data["samples"]:
                        check_score(row, self.vocabulary)
                        self.canonical[backend, corpus, length, row["position"]] = row
                    if backend == "cuda_canonical":
                        self.transfers[1].add(length, len(positions), len(groups))
        require(self.report["canonical"] == expected, "独立参照集合不符")

    def teacher(self):
        require([row["id"] for row in self.report["teacher_forcing"]] == [row["id"] for row in self.plan],
                "teacher-forcing 结果缺失、重复或乱序")
        for descriptor, entry in zip(self.plan, self.report["teacher_forcing"]):
            identity = descriptor["id"]
            require(entry["file"] == f"teacher/{identity}.json", "teacher-forcing 路径不符")
            data = self.load(entry["file"])
            require(data["schema_version"] == 1
                    and all(data[key] == value for key, value in descriptor.items()), "teacher-forcing 描述不符")
            corpus, length, chunk, sequences = (descriptor[key] for key in ("corpus", "length", "chunk", "sequences"))
            groups = batches(self.contract, corpus, length, chunk, sequences)
            selected = [(b, i, row) for b, batch in enumerate(groups) for i, row in enumerate(batch) if row[3]]
            require(len(data["comparisons"]) == len(selected) * 3, "teacher-forcing 比较数量不符")
            passed = True
            for index, (batch_id, input_index, row) in enumerate(selected):
                _, position, sequence, _ = row
                actual = None
                for ref_index, backend in enumerate(REFERENCES):
                    record = data["comparisons"][index * 3 + ref_index]
                    ref_corpus = (corpus + sequence) % len(self.contract["corpus"])
                    expected_identity = dict(case=identity, reference=backend, corpus=ref_corpus, sequence=sequence,
                                             position=position, batch=batch_id, input_index=input_index,
                                             batch_tokens=len(groups[batch_id]))
                    require(all(record[key] == value for key, value in expected_identity.items()), "比较行身份不符")
                    expected = self.canonical[backend, ref_corpus, length, position]
                    current_passed = compare_record(record, self.contract["thresholds"], expected, actual)
                    require(record["actual_argmax"] < self.vocabulary, "实际 argmax 超出词表")
                    actual = dict(sha256=record["actual_sha256"], token=record["actual_argmax"])
                    passed = passed and current_passed
                    self.metrics.add(record, True)
            require(data["passed"] is passed and entry == dict(id=identity, file=entry["file"],
                    sampled_rows=len(selected), passed=passed), "teacher-forcing 用例状态不符")
            self.metrics.totals["teacher_cases"] += 1
            self.metrics.totals["teacher_rows"] += len(selected)
            transfer = self.transfers[sequences]
            transfer.add(length * sequences, len(selected), len(groups))
            transfer.check(data["after"], self.configurations[sequences]["before"], [length] * sequences, self.vocabulary)
            if (corpus, length, chunk, sequences) == (0, 1536, 33, 4):
                self.timing(groups, identity, len(selected))
                transfer.add(length * sequences, len(selected), len(groups))

    def timing(self, groups, identity, rows):
        value = self.report["timing"]
        require(value["case"] == identity and value["bitwise_rows"] == rows == 40 and value["passed"] is True,
                "计时开关位级检查不符")
        require(len(value["batches"]) == len(groups), "计时开关批次数量不符")
        for index, (batch, record) in enumerate(zip(groups, value["batches"])):
            require(record["batch"] == index and record["input_rows"] == len(batch)
                    and record["logits_rows"] == sum(row[3] for row in batch) and record["bitwise_equal"] is True
                    and finite(record["device_elapsed_ms"]) and record["device_elapsed_ms"] >= 0,
                    "计时开关批次检查不符")

    def golden(self):
        expected = []
        for sequences in (1, 4):
            for c in self.contract["stable_greedy"]:
                expected.append(dict(max_sequences=sequences, text=c["text"],
                                     expected=c["expected_token_ids"], actual=c["expected_token_ids"], passed=True))
                self.transfers[sequences].add(len(c["input_token_ids"]) + 7, 8, 8, False)
        require(self.report["golden"] == expected, "短样例的 8-token 金标准不符")

    def generation(self):
        entries = []
        for corpus, c in enumerate(self.contract["corpus"]):
            for length in (16, 128, 1536):
                identity = f"{c['id']}-l{length}-g32"
                file = f"generation/cuda-{identity}.json"
                data = self.load(file)
                require(data["schema_version"] == 1 and data["id"] == identity and data["corpus"] == corpus
                        and data["prompt_tokens"] == length and data["output_tokens"] == 32
                        and data["final_kv_tokens"] == length + 31, "自然生成用例身份不符")
                references = {}
                for backend in REFERENCES[:2]:
                    reference = self.load(f"generation/{backend}-{identity}.json")
                    require(reference["schema_version"] == 1 and reference["backend"] == backend
                            and reference["corpus"] == corpus and reference["prompt_tokens"] == length
                            and reference["output_tokens"] == 32 and reference["final_kv_tokens"] == length + 31
                            and len(reference["samples"]) == 32, "自然生成参照身份不符")
                    references[backend] = reference["samples"]
                first, passed = generation_rows(data, references, self.contract, self.metrics)
                entries.append(dict(id=identity, file=file, passed=passed, first_divergence=first))
                self.transfers[4].add(length + 31, 32, math.ceil(length / 128) + 31)
                self.transfers[4].check(data["after"], self.configurations[4]["before"],
                                        [length + 31, 0, 0, 0], self.vocabulary)
        require(self.report["generation"] == entries, "自然生成结果集合不符")

    def boundary(self):
        value = self.report["boundary"]
        require(value["slot"] == 3 and value["length"] == 2048 and value["passed"] is True
                and value["reuse_bitwise_equal"] is True and value["fresh"] == value["reuse"],
                "2048-token 边界或 clear 复用检查不符")
        for name in ("last", "fresh", "reuse"):
            check_score(value[name], self.vocabulary)
        require(value["before_rejected_append"] == value["after_rejected_append"], "拒绝追加改变了 Runtime 状态")
        self.transfers[4].add(16 + 2048, 2, 17)
        self.transfers[4].check(value["before_rejected_append"], self.configurations[4]["before"],
                                [0, 0, 0, 2048], self.vocabulary)
        self.transfers[4].add(16, 1, 1)

    def weights(self):
        weights = self.load("weight-plan.json")
        unique = {}
        source_counts = dict(F32=0, Q8_0=0)
        offset = 0
        for row in weights:
            require(row["device_dtype"] == "F32" and digest(row["effective_sha256"])
                    and len(row["shape"]) == 2 and all(integer(n, 1) for n in row["shape"])
                    and row["bytes"] == math.prod(row["shape"]) * 4, "权重格式不符")
            if row["alias_of"]:
                require(row["alias_of"] in unique and all(row[key] == unique[row["alias_of"]][key]
                        for key in ("shape", "offset", "bytes", "source_dtype", "effective_sha256")), "权重别名不符")
            else:
                require(row["name"] not in unique and row["offset"] == offset and row["source_dtype"] in source_counts,
                        "权重范围重复或不连续")
                unique[row["name"]] = row
                source_counts[row["source_dtype"]] += 1
                offset += row["bytes"]
        require(len(weights) == 311 and len(unique) == 310
                and source_counts == self.contract["arithmetic"]["source_tensor_counts"]
                and offset == self.configurations[4]["before"]["resident"]["weights_bytes"], "权重清单汇总不符")

    def run(self):
        require(self.report["schema_version"] == self.summary["schema_version"] == 1
                and self.report["scope"] == self.summary["scope"] == SCOPE
                and self.report["complete"] is True and self.summary["complete"] is True, "报告未完成或范围无效")
        reference_settings(self.report, self.summary)
        self.inputs()
        self.configuration()
        self.references()
        self.teacher()
        self.golden()
        self.generation()
        self.boundary()
        self.weights()
        for sequences, config in self.configurations.items():
            self.transfers[sequences].check(config["after"], config["before"], [0] * sequences, self.vocabulary)
        require(self.report["expected"] == dict(teacher_cases=240, teacher_rows=3920, teacher_comparisons=11760,
                canonical_cases=60, generation_cases=12, golden_cases=6), "预期覆盖数不符")
        for report in (self.report, self.summary):
            require(report["totals"] == self.metrics.totals and report["extrema"] == self.metrics.extrema,
                    "汇总指标与逐行记录不符")
            require(report["first_numeric_failure"] == self.metrics.first_failure
                    and report["first_argmax_divergence"] == self.metrics.first_divergence, "首次失败或分歧摘要不符")
        passed = self.metrics.totals["numeric_failures"] == 0
        require(all(report["passed"] is passed and report["status"] == ("passed" if passed else "failed")
                    for report in (self.report, self.summary)), "总体状态与逐行验收不符")
        require(self.summary["complete_gpu_model"] is passed and self.summary["full_corpus_contract"] is passed
                and self.summary["performance_baseline"] is False and self.summary["gpu_serving"] is False,
                "模型、性能或 Serving 范围声明不符")
        require(self.summary["model_sha256"] == self.contract["model"]["sha256"]
                and self.summary["reference_sha256"] == self.contract["reference"]["sha256"]
                and self.summary["golden_cases"] == 6 and self.summary["generation_cases"] == 12, "摘要身份或计数不符")
        return dict(schema_version=1, status="passed" if passed else "numeric_failed", passed=passed,
                    scope=SCOPE, totals=self.metrics.totals, extrema=self.metrics.extrema,
                    by_reference=self.metrics.by_reference, first_numeric_failure=self.metrics.first_failure,
                    first_argmax_divergence=self.metrics.first_divergence, gpu_serving=False, performance_baseline=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=Path(__file__).resolve().parents[1] / "tests/data/qwen3_validation_cases.json")
    options = parser.parse_args()
    try:
        result = Audit(options.directory, read(options.contract)).run()
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result["passed"] else 1
    except (ValidationError, OSError, KeyError, IndexError, TypeError, ValueError) as error:
        print(f"CUDA 数值证据复核失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
