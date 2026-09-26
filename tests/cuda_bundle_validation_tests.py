"""完整 CUDA 证据包的语义、文件闭合、来源身份和无覆盖发布检查。"""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cuda_evidence_bundle as audit


def invalid(function):
    try:
        function()
    except (audit.model.ValidationError, OSError, KeyError, ValueError):
        return
    raise AssertionError("不完整或无效证据包未被拒绝")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_state():
    return dict(files=[
        dict(path="scripts/cuda_evidence_bundle.py", sha256=audit.sha(Path(audit.__file__))),
        *(dict(path="scripts/" + name, sha256=audit.sha(audit.HELPERS / name)) for name in audit.HELPER_FILES),
    ])


def input_directories(root):
    result = {}
    for name in audit.COMPONENTS:
        directory = root / name
        write(directory / "fixture.json", dict(scope="deterministic_fixture_not_real_gpu"))
        result[name] = directory
    write(result["validation"] / "source-state.json", source_state())
    return result


def component_results(root):
    directories = input_directories(root)
    model_result = dict(summary=dict(status="measurement_inconclusive", reports=70, measured_repetitions=2520))
    micro_result = dict(summary=dict(case_count=375, independent_trials=5))
    profiler_result = dict(nsys=dict(forwards=585), ncu=dict(profiled_kernel_count=1), overhead={})
    write(directories["model"] / "summary.json", model_result["summary"])
    write(directories["micro"] / "summary.json", micro_result["summary"])
    for name, key in (("nsys-summary.json", "nsys"), ("ncu-selected-kernel.json", "ncu"),
                      ("profiler-overhead.json", "overhead")):
        write(directories["profiler"] / name, profiler_result[key])
    return directories, model_result, micro_result, profiler_result


def derived_summaries_are_recomputed_even_with_valid_file_hashes():
    targets = (("model", "summary.json"), ("micro", "summary.json"),
               ("profiler", "nsys-summary.json"), ("profiler", "ncu-selected-kernel.json"),
               ("profiler", "profiler-overhead.json"))
    for component, name in targets:
        with tempfile.TemporaryDirectory() as directory:
            directories, model_result, micro_result, profiler_result = component_results(Path(directory))
            write(directories[component] / name, dict(status="fabricated"))
            with patch.object(audit.model, "validate_bundle", return_value=model_result), \
                 patch.object(audit.micro, "validate_bundle", return_value=micro_result), \
                 patch.object(audit.profiler, "validate_bundle", return_value=profiler_result), \
                 patch.object(audit, "powershell_verify", return_value=dict(exit_code=0)):
                try:
                    audit.validate_components(directories)
                except audit.model.ValidationError as error:
                    assert "摘要" in str(error)
                else:
                    raise AssertionError("重算不符的派生摘要未被拒绝")


def fixture(root):
    root.mkdir()
    for name, destination in audit.COMPONENTS.items():
        write(root / destination / "fixture.json", dict(scope="deterministic_fixture_not_real_gpu"))
    write(root / audit.COMPONENTS["validation"] / "source-state.json", source_state())
    shutil.copyfile(Path(audit.__file__), root / "verify.py")
    (root / "verification").mkdir()
    for name in audit.HELPER_FILES:
        shutil.copyfile(audit.HELPERS / name, root / "verification" / name)
    (root / "README.md").write_text("确定性归档 fixture，不是 GPU 实测。\n", encoding="utf-8")
    refresh_manifest(root)


def refresh_manifest(root):
    entries = audit.files(root)
    entries.pop("bundle-manifest.json", None)
    write(root / "bundle-manifest.json", dict(
        schema_version=1, benchmark=audit.BENCHMARK, purpose="archive_revalidation", spec_id="CUDA-VS-001",
        components=audit.COMPONENTS, execution_dependencies_included=False, artifacts=list(entries.values())))


def closed_file_index_is_verified():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "bundle"
        fixture(root)
        with patch.object(audit, "validate_components", return_value=dict(status="passed")):
            result = audit.verify(root)
        assert result["status"] == "passed" and result["artifact_count"] == len(audit.files(root))


def missing_tampered_or_unregistered_files_are_rejected():
    for change in ("missing", "bytes", "unregistered", "omitted_index"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            fixture(root)
            target = root / audit.COMPONENTS["model"] / "fixture.json"
            if change == "missing":
                target.unlink()
            elif change == "bytes":
                target.write_text("{}\n", encoding="utf-8")
            elif change == "unregistered":
                (root / "unregistered.txt").write_text("extra\n", encoding="utf-8")
            else:
                manifest = audit.read(root / "bundle-manifest.json")
                manifest["artifacts"] = [item for item in manifest["artifacts"] if item["path"] != target.relative_to(root).as_posix()]
                write(root / "bundle-manifest.json", manifest)
            with patch.object(audit, "validate_components", return_value=dict(status="passed")):
                invalid(lambda: audit.verify(root))


def verifier_source_is_checked_even_after_rehashing_index():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "bundle"
        fixture(root)
        with (root / "verification/analyze_cuda_benchmark.py").open("a", encoding="utf-8") as stream:
            stream.write("\n# 确定性来源篡改。\n")
        refresh_manifest(root)
        with patch.object(audit, "validate_components", return_value=dict(status="passed")):
            invalid(lambda: audit.verify(root))


def duplicate_paths_boolean_versions_and_omitted_components_are_rejected():
    for change in ("duplicate", "schema", "component"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            fixture(root)
            manifest = audit.read(root / "bundle-manifest.json")
            if change == "duplicate":
                manifest["artifacts"].append(deepcopy(manifest["artifacts"][0]))
            elif change == "schema":
                manifest["schema_version"] = True
            else:
                manifest["components"].pop("profiler")
            write(root / "bundle-manifest.json", manifest)
            with patch.object(audit, "validate_components", return_value=dict(status="passed")):
                invalid(lambda: audit.verify(root))


def model_files_and_symlinks_are_not_exported():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / ".gitattributes").write_text("*.txt -text\n", encoding="ascii")
        assert set(audit.files(root)) == {".gitattributes"}
        (root / "weights.gguf").write_bytes(b"not actual weights")
        invalid(lambda: audit.files(root))
        (root / "weights.gguf").unlink()
        (root / "data.json").write_text("{}\n", encoding="utf-8")
        (root / "link.json").symlink_to(root / "data.json")
        invalid(lambda: audit.files(root))


def existing_archive_or_checksum_is_preserved():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "evidence.zip"
        output.write_bytes(b"old archive")
        invalid(lambda: audit.export_bundle({}, output))
        assert output.read_bytes() == b"old archive"
        output.unlink()
        checksum = root / "evidence.zip.sha256"
        checksum.write_bytes(b"old checksum")
        invalid(lambda: audit.export_bundle({}, output))
        assert checksum.read_bytes() == b"old checksum" and not output.exists()


def export_publication_is_atomic_and_sources_are_preserved():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sources = input_directories(root / "inputs")
        before = {name: audit.files(path) for name, path in sources.items()}
        completed = subprocess.CompletedProcess([], 0, '{"status":"passed"}\n', "")
        with patch.object(audit, "validate_components", return_value=dict(status="passed")), \
             patch.object(audit.subprocess, "run", return_value=completed):
            result = audit.export_bundle(sources, root / "evidence.zip")
        assert result["status"] == "passed" and result["source_files_unchanged"] is True
        assert {name: audit.files(path) for name, path in sources.items()} == before
        assert result["sha256"] == audit.sha(root / "evidence.zip")
        assert (root / "evidence.zip.sha256").read_text(encoding="ascii").startswith(result["sha256"])
        with zipfile.ZipFile(root / "evidence.zip") as archive:
            assert "bundle-manifest.json" in archive.namelist()
            assert set("verification/" + name for name in audit.HELPER_FILES) <= set(archive.namelist())


def failed_publication_rolls_back_only_its_own_files():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sources = input_directories(root / "inputs")
        preserved = root / "preserved.txt"
        preserved.write_bytes(b"unchanged")
        completed = subprocess.CompletedProcess([], 0, '{"status":"passed"}\n', "")
        link, calls = os.link, 0

        def fail_second(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("确定性摘要发布故障")
            link(source, destination)

        with patch.object(audit, "validate_components", return_value=dict(status="passed")), \
             patch.object(audit.subprocess, "run", return_value=completed), \
             patch.object(audit.os, "link", side_effect=fail_second):
            invalid(lambda: audit.export_bundle(sources, root / "evidence.zip"))
        assert not (root / "evidence.zip").exists() and not (root / "evidence.zip.sha256").exists()
        assert preserved.read_bytes() == b"unchanged"


if __name__ == "__main__":
    tests = [derived_summaries_are_recomputed_even_with_valid_file_hashes,
             closed_file_index_is_verified, missing_tampered_or_unregistered_files_are_rejected,
             verifier_source_is_checked_even_after_rehashing_index,
             duplicate_paths_boolean_versions_and_omitted_components_are_rejected,
             model_files_and_symlinks_are_not_exported, existing_archive_or_checksum_is_preserved,
             export_publication_is_atomic_and_sources_are_preserved,
             failed_publication_rolls_back_only_its_own_files]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"{len(tests)}/{len(tests)} tests passed")
