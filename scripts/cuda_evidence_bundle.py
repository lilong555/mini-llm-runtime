"""CUDA-VS-001 完整证据包的可迁移导出与离线复核，不执行模型或 GPU 工具。"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HELPERS = HERE / "verification" if (HERE / "verification").is_dir() else HERE
sys.path.insert(0, str(HELPERS))
import analyze_cuda_benchmark as model
import analyze_cuda_micro as micro
import analyze_cuda_profiler as profiler


BENCHMARK = "minillm-cuda-evidence-bundle"
COMPONENTS = {
    "model": "benchmarks/results/cuda-model-baseline",
    "micro": "benchmarks/results/cuda-micro-baseline",
    "numerical": "benchmarks/results/validation/cuda-micro",
    "profiler": "benchmarks/results/cuda-model-profiler",
    "validation": "benchmarks/results/validation/cuda-profiler",
}
HELPER_FILES = ("analyze_cuda_benchmark.py", "analyze_cuda_micro.py", "analyze_cuda_profiler.py")
ALLOWED_SUFFIXES = {".json", ".txt", ".md", ".csv", ".xml", ".zip", ".py", ".ps1", ".sha256"}
require = model.require
read = model.read
sha = model.sha


def files(directory):
    directory = directory.resolve()
    result = {}
    for path in sorted(directory.rglob("*")):
        if "__pycache__" in path.relative_to(directory).parts or path.suffix == ".pyc":
            continue
        require(not path.is_symlink(), "完整证据包不能依赖符号链接：" + str(path))
        if path.is_dir():
            continue
        require(path.is_file() and (path.suffix in ALLOWED_SUFFIXES or path.name == ".gitattributes"),
                "完整包不接受模型、二进制或其他未登记类型：" + str(path))
        name = path.relative_to(directory).as_posix()
        model.relative_path(name)
        result[name] = dict(path=name, sha256=sha(path), size_bytes=path.stat().st_size)
    require(result, "证据目录为空")
    return result


def powershell_verify(directory):
    executable = shutil.which("pwsh")
    require(executable is not None, "完整证据复核需要 PowerShell 7")
    entry = model.artifact(directory, "verify.ps1")
    command = [executable, "-NoProfile", "-File", str(entry), "-Directory", str(directory)]
    result = subprocess.run(command, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    require(result.returncode == 0, "归档验证失败：" + str(directory) + "\n" + result.stdout)
    return dict(exit_code=result.returncode, output=result.stdout)


def validate_components(directories):
    model_result = model.validate_bundle(directories["model"])
    micro_result = micro.validate_bundle(directories["micro"])
    require(model.identical(read(directories["model"] / "summary.json"), model_result["summary"]),
            "模型摘要与原始样本的重新计算结果不同")
    require(model.identical(read(directories["micro"] / "summary.json"), micro_result["summary"]),
            "微基准摘要与原始样本的重新计算结果不同")
    require(model_result["summary"]["status"] in ("measured", "measurement_inconclusive"),
            "模型基线存在正确性后续事项，不能关闭 M1")
    numeric_check = powershell_verify(directories["numerical"])
    tool_check = powershell_verify(directories["validation"])
    profiler_result = profiler.validate_bundle(directories["profiler"])
    for name, key in (("nsys-summary.json", "nsys"), ("ncu-selected-kernel.json", "ncu"),
                      ("profiler-overhead.json", "overhead")):
        require(model.identical(read(directories["profiler"] / name), profiler_result[key]),
                "Profiler 派生摘要与原始证据不同：" + name)
    baseline = read(directories["model"] / "manifest.json")
    micro_manifest = read(directories["micro"] / "manifest.json")
    profile = read(directories["profiler"] / "manifest.json")
    numeric = read(directories["numerical"] / "real-model/validation-summary.json")
    numeric_environment = read(directories["numerical"] / "environment.json")
    tool_environment = read(directories["validation"] / "environment.json")
    require(profile["baseline"]["manifest_sha256"] == sha(directories["model"] / "manifest.json")
            and profile["baseline"]["run_id"] == baseline["run_id"]
            and profile["binary"]["sha256"] == baseline["binary"]["sha256"],
            "Profiler 与完整包中的模型基线不同")
    require(sha(directories["profiler"] / "baseline-summary.json") == sha(directories["model"] / "summary.json")
            and sha(directories["profiler"] / "baseline-validation-summary.json")
            == sha(directories["model"] / "validation-summary.json"), "关联基线摘要与完整包内容不同")
    require(sha(directories["model"] / "validation-summary.json")
            == sha(directories["numerical"] / "real-model/validation-summary.json"),
            "模型基线没有关联完整包中的数值验证")
    require(baseline["numerical_evidence"]["validation_binary_sha256"]
            == numeric_environment["full_validation_binary_sha256"]
            == tool_environment["full_validation_binary_sha256"], "完整包的数值编译身份不一致")
    require(tool_environment["runtime_benchmark_binary_sha256"] == baseline["binary"]["sha256"]
            and numeric_environment["micro_binary_sha256"] == micro_manifest["binary"]["sha256"],
            "工具验收、微基准或模型基准的二进制身份不符")
    profile_state = read(directories["profiler"] / "source-state.json")
    numeric_state = read(directories["numerical"] / "source-state.json")
    tool_state = read(directories["validation"] / "source-state.json")

    def runtime_sources(state):
        return {item["path"]: item["sha256"] for item in state["files"]
                if item["path"].startswith(("include/minillm/", "src/minillm/"))}

    require(runtime_sources(profile_state) == runtime_sources(numeric_state) == runtime_sources(tool_state),
            "完整包各阶段的模型 Runtime 源码不同")
    return dict(status="passed", spec_id="CUDA-VS-001", model_performance_status=model_result["summary"]["status"],
                model_reports=model_result["summary"]["reports"],
                model_measured_repetitions=model_result["summary"]["measured_repetitions"],
                micro_cases=micro_result["summary"]["case_count"],
                micro_trials=micro_result["summary"]["independent_trials"],
                full_numeric_comparisons=numeric["totals"]["teacher_comparisons"] + numeric["totals"]["generation_comparisons"],
                profiler_forwards=profiler_result["nsys"]["forwards"],
                ncu_kernels=profiler_result["ncu"]["profiled_kernel_count"],
                numerical_archive_verification=numeric_check, tool_archive_verification=tool_check,
                mandatory_data_path_gates="passed", gpu_serving=False, gpu_paged_attention=False,
                positive_performance_is_not_a_completion_requirement=True,
                decision="V2-M1 的证据门禁完成；下一阶段为 V2-M2，性能不确定项继续保留")


def verify(directory):
    directory = directory.resolve()
    manifest = read(model.artifact(directory, "bundle-manifest.json"))
    require(model.identical(manifest["schema_version"], 1) and manifest["benchmark"] == BENCHMARK
            and manifest["purpose"] == "archive_revalidation" and manifest["spec_id"] == "CUDA-VS-001"
            and manifest["components"] == COMPONENTS and manifest["execution_dependencies_included"] is False,
            "完整 CUDA 证据包身份或组成不符")
    expected = {}
    for item in manifest["artifacts"]:
        name = item["path"]
        require(name != "bundle-manifest.json" and name not in expected and model.digest(item["sha256"])
                and model.integer(item["size_bytes"]), "完整包产物索引包含重复路径或无效字段")
        path = model.artifact(directory, name)
        require(sha(path) == item["sha256"] and path.stat().st_size == item["size_bytes"], "完整包产物内容不符：" + name)
        expected[name] = item
    actual = files(directory)
    original_manifest = actual.pop("bundle-manifest.json")
    require(set(actual) == set(expected), "完整包有未登记或缺失的文件")
    required = {"verify.py", "README.md", *(f"verification/{name}" for name in HELPER_FILES)}
    require(required <= set(expected), "完整包缺少独立复核工具")
    directories = {key: directory / value for key, value in COMPONENTS.items()}
    for key, path in directories.items():
        require(path.is_dir(), "完整包缺少阶段：" + key)
    tool_state = read(directories["validation"] / "source-state.json")
    source_files = {item["path"]: item["sha256"] for item in tool_state["files"]}
    for bundled, source in (("verify.py", "scripts/cuda_evidence_bundle.py"),
                            *((f"verification/{name}", f"scripts/{name}") for name in HELPER_FILES)):
        require(sha(directory / bundled) == source_files[source], "完整包复核工具不属于工具验收源码")
    result = validate_components(directories)
    require(files(directory) == {**expected, "bundle-manifest.json": original_manifest},
            "归档复核改变了包内文件")
    result["artifact_count"] = len(expected) + 1
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def export_bundle(directories, output):
    output = output.resolve()
    checksum = Path(str(output) + ".sha256")
    require(output.suffix == ".zip" and output.parent.is_dir(), "导出目标必须为已有目录内的新 ZIP")
    require(not output.exists() and not output.is_symlink() and not checksum.exists() and not checksum.is_symlink(),
            "不能覆盖已有完整证据包或摘要")
    directories = {name: path.resolve() for name, path in directories.items()}
    require(set(directories) == set(COMPONENTS), "导出缺少必需阶段目录")
    frozen = {name: files(path) for name, path in directories.items()}
    validate_components(directories)
    with tempfile.TemporaryDirectory(prefix="cuda-bundle-", dir=output.parent) as temporary:
        root = Path(temporary)
        stage = root / "bundle"
        stage.mkdir()
        for name, source in directories.items():
            for item in frozen[name].values():
                target = stage / COMPONENTS[name] / item["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(model.artifact(source, item["path"]), target)
                require(sha(target) == item["sha256"], "导出期间原始证据发生变化：" + str(target))
        shutil.copyfile(Path(__file__), stage / "verify.py")
        (stage / "verification").mkdir()
        for name in HELPER_FILES:
            shutil.copyfile(HELPERS / name, stage / "verification" / name)
        (stage / "README.md").write_text(
            "# CUDA-VS-001 完整证据包\n\n"
            "包含独立的模型 A/A/异构基线、真实形状微基准、完整数值与回归验证、完整模型 Profiler。\n"
            "各阶段的源码状态、二进制身份、原始样本和负结果分别保留。\n\n"
            "## 复核\n\n"
            "需要 Python 3.10+ 与 PowerShell 7，不需要原采集路径、模型、编译产物、CUDA 或 Nsight。\n\n"
            "```bash\npython3 -B verify.py --directory .\n```\n\n"
            "完整包根目录的复核器验证所有阶段及其关联；子归档中的历史复核工具同时保留原始身份。\n"
            "NSys/NCU 原始报告和 SQLite 位于 Profiler 子归档的 `profiler-raw.zip`，索引包含逐文件 SHA-256。\n"
            "模型权重、可执行文件和第三方依赖未装入归档，复核不是原二进制重跑或可信执行证明。\n"
            "无 Profiler 模型时延、外部诊断、kernel 时间、逻辑流量和硬件计数器不能混同。\n"
            "GPU Serving、GPU prefix sharing 与自有 PagedAttention 不在本包完成声明中。\n",
            encoding="utf-8")
        artifacts = files(stage)
        write_json(stage / "bundle-manifest.json", dict(
            schema_version=1, benchmark=BENCHMARK, spec_id="CUDA-VS-001", purpose="archive_revalidation",
            components=COMPONENTS, artifacts=list(artifacts.values()), verification_entry="verify.py",
            execution_dependencies_included=False,
            requirements=dict(python=">=3.10", powershell=">=7", nvidia_tools_required=False)))
        verify(stage)
        pending = root / "bundle.zip"
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name in files(stage):
                archive.write(stage / name, name)
        roundtrip = root / "roundtrip"
        roundtrip.mkdir()
        with zipfile.ZipFile(pending) as archive:
            archive.extractall(roundtrip)
        command = [sys.executable, "-B", str(roundtrip / "verify.py"), "--directory", str(roundtrip)]
        checked = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                 env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        require(checked.returncode == 0, "独立目录复核失败：\n" + checked.stdout + checked.stderr)
        for name, directory in directories.items():
            require(files(directory) == frozen[name], "导出或复核改变了原始证据：" + name)
        digest = sha(pending)
        pending_checksum = root / "bundle.sha256"
        pending_checksum.write_text(digest + "  " + output.name + "\n", encoding="ascii")
        published = []
        try:
            # 同文件系统硬链接以原子、不可覆盖的方式发布；失败只回收本次已发布的文件。
            os.link(pending, output)
            published.append(output)
            os.link(pending_checksum, checksum)
            published.append(checksum)
        except OSError:
            for path in reversed(published):
                path.unlink()
            raise
        return dict(status="passed", archive=str(output), sha256=digest, size_bytes=output.stat().st_size,
                    independent_directory_revalidation=dict(command=command, exit_code=checked.returncode,
                                                            result=json.loads(checked.stdout)),
                    source_files_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--directory", type=Path)
    mode.add_argument("--export", type=Path)
    for name in COMPONENTS:
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    try:
        if args.directory:
            result = verify(args.directory)
        else:
            values = {name: getattr(args, name) for name in COMPONENTS}
            require(all(values.values()), "导出必须指定 model、micro、numerical、profiler、validation 目录")
            result = export_bundle(values, args.export)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (model.ValidationError, KeyError, ValueError, TypeError, OSError, zipfile.BadZipFile) as error:
        print(f"CUDA 完整证据包复核失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
