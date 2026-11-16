"""Parallel Q-Chem runner"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class QChemJobResult:
    input_path: Path
    output_path: Path
    returncode: int
    stdout: str
    stderr: str
    skipped: bool = False

    @property
    def failed(self) -> bool:
        return (not self.skipped) and self.returncode != 0


class QChemJobError(RuntimeError):
    """Raised when a required Q-Chem job exits unsuccessfully."""


PROVENANCE_SCHEMA_VERSION = 1
NORMAL_TERMINATION_TEXT = "Thank you very much for using Q-Chem"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qchem_provenance_path(output_path: str | Path) -> Path:
    output = Path(output_path)
    return output.with_name(output.name + ".provenance.json")


def qchem_output_has_normal_termination(output_path: str | Path) -> bool:
    path = Path(output_path)
    if not path.exists():
        return False
    return NORMAL_TERMINATION_TEXT in path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def qchem_output_matches_input(input_path: str | Path, output_path: str | Path) -> bool:
    """Return whether a completed output has matching cryptographic provenance."""
    input_file = Path(input_path)
    output_file = Path(output_path)
    provenance_file = qchem_provenance_path(output_file)
    if not input_file.exists() or not output_file.exists() or not provenance_file.exists():
        return False
    if not qchem_output_has_normal_termination(output_file):
        return False
    try:
        record = json.loads(provenance_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return (
        record.get("schema_version") == PROVENANCE_SCHEMA_VERSION
        and record.get("input_sha256") == file_sha256(input_file)
        and record.get("output_sha256") == file_sha256(output_file)
    )


def write_qchem_provenance(input_path: str | Path, output_path: str | Path) -> Path:
    """Write provenance for a known-complete input/output pair."""
    input_file = Path(input_path).resolve()
    output_file = Path(output_path).resolve()
    if not qchem_output_has_normal_termination(output_file):
        raise ValueError(f"Q-Chem output did not terminate normally: {output_file}")
    provenance_file = qchem_provenance_path(output_file)
    temporary = provenance_file.with_name(provenance_file.name + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "input_path": str(input_file),
                "output_path": str(output_file),
                "input_sha256": file_sha256(input_file),
                "output_sha256": file_sha256(output_file),
                "normal_termination": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(provenance_file)
    return provenance_file


def run_qchem_filelist(
    filelist: str | Path,
    *,
    processes: int,
    command_template: str,
    skip_existing_outputs: bool = True,
    skip_failed_jobs: bool = False,
    verbose: bool = False,
) -> list[QChemJobResult]:
    """Run Q-Chem jobs listed in a file, with bounded parallelism."""
    filelist_path = Path(filelist).resolve()
    if processes < 1:
        raise ValueError("processes must be at least 1")
    input_paths = _read_filelist(filelist_path)

    skipped_results = [
        QChemJobResult(input_path, input_path.with_suffix(".out"), 0, "", "", skipped=True)
        for input_path in input_paths
        if skip_existing_outputs
        and qchem_output_matches_input(input_path, input_path.with_suffix(".out"))
    ]
    skipped_inputs = {result.input_path for result in skipped_results}
    jobs_to_run = [input_path for input_path in input_paths if input_path not in skipped_inputs]
    invalid_existing_count = sum(
        input_path.with_suffix(".out").exists() and input_path not in skipped_inputs
        for input_path in input_paths
    )

    if verbose:
        print(
            f"Q-Chem runner: loaded {len(input_paths)} input file(s) from {filelist_path}",
            flush=True,
        )
        print(
            f"Q-Chem runner: {len(jobs_to_run)} job(s) to run, "
            f"{len(skipped_results)} validated existing output(s) skipped",
            flush=True,
        )
        if invalid_existing_count:
            print(
                f"Q-Chem runner: {invalid_existing_count} existing output(s) are stale, "
                "modified, incomplete, or lack provenance and will be rerun",
                flush=True,
            )
        if skipped_results and not jobs_to_run:
            print(
                "Q-Chem runner: all outputs have matching input hashes and normal termination; set "
                "qchem.skip_existing_outputs: false or delete .out files to rerun.",
                flush=True,
            )

    results: list[QChemJobResult] = list(skipped_results)
    if not jobs_to_run:
        return results

    workers = min(processes, len(jobs_to_run))
    core_budget = int(os.environ.get("QCHEM_BATCH_CORE_BUDGET", "0"))
    if core_budget:
        if core_budget < 1:
            raise ValueError("QCHEM_BATCH_CORE_BUDGET must be positive")
        workers = min(workers, core_budget)
        # Count unfinished jobs: a resumed batch may have only one calculation
        # left even when its original file list contained many geometries.
        threads = core_budget if workers == 1 else 1
        words = shlex.split(command_template)
        if "-nt" in words:
            words[words.index("-nt") + 1] = str(threads)
        else:
            words[1:1] = ["-nt", str(threads)]
        command_template = shlex.join(words)
        if verbose:
            print(
                f"Q-Chem: up to {workers} concurrent jobs, -nt {threads} each.",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _run_one_job,
                input_path,
                command_template=command_template,
                skip_existing_outputs=False,
            )
            for input_path in jobs_to_run
        ]
        completed_count = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed_count += 1
            if result.returncode != 0:
                if skip_failed_jobs:
                    if verbose:
                        print(
                            f"Q-Chem runner: job failed but continuing: "
                            f"{result.input_path} (return code {result.returncode})",
                            flush=True,
                        )
                else:
                    raise QChemJobError(format_qchem_failure(result))
            if verbose:
                should_report = (
                    completed_count == 1
                    or completed_count == len(jobs_to_run)
                    or completed_count % max(1, min(25, len(jobs_to_run))) == 0
                )
                if should_report:
                    print(
                        f"Q-Chem runner: completed {completed_count}/{len(jobs_to_run)} job(s)",
                        flush=True,
                    )
    return results


def format_qchem_failure(result: QChemJobResult) -> str:
    output_tail = "(missing)"
    if result.output_path.exists():
        output_tail = tail_text(result.output_path, max_lines=40)
    return (
        "Q-Chem job failed.\n"
        f"  input: {result.input_path}\n"
        f"  output: {result.output_path}\n"
        f"  return code: {result.returncode}\n"
        f"  stderr tail:\n{indent_block(tail_string(result.stderr))}\n"
        f"  stdout tail:\n{indent_block(tail_string(result.stdout))}\n"
        f"  output tail:\n{indent_block(output_tail)}\n"
        "For noncritical transverse normal-mode scans, set "
        "`qchem.skip_failed_jobs: true` to continue and omit missing/unparsable points."
    )


def tail_string(text: str, *, max_lines: int = 20) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:]) if lines else "(empty)"


def tail_text(path: Path, *, max_lines: int = 40) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:]) if lines else "(empty)"


def indent_block(text: str) -> str:
    return "\n".join(f"    {line}" for line in text.splitlines())


def _read_filelist(filelist_path: Path) -> list[Path]:
    paths: list[Path] = []
    base_dir = filelist_path.parent
    for raw_line in filelist_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not line.endswith(".in"):
            line = f"{line}.in"
        path = Path(line)
        if not path.is_absolute():
            path = base_dir / path
        paths.append(path)
    if not paths:
        raise ValueError(f"No Q-Chem input files are listed in {filelist_path}")
    return paths


def _run_one_job(
    input_path: Path,
    *,
    command_template: str,
    skip_existing_outputs: bool,
) -> QChemJobResult:
    output_path = input_path.with_suffix(".out")
    if skip_existing_outputs and qchem_output_matches_input(input_path, output_path):
        return QChemJobResult(input_path, output_path, 0, "", "", skipped=True)

    provenance_path = qchem_provenance_path(output_path)
    if provenance_path.exists():
        provenance_path.unlink()

    command = command_template.format(input=str(input_path), output=str(output_path))
    completed = subprocess.run(
        shlex.split(command),
        cwd=str(input_path.parent),
        text=True,
        capture_output=True,
        check=False,
    )
    returncode = completed.returncode
    stderr = completed.stderr
    if returncode == 0 and not qchem_output_has_normal_termination(output_path):
        returncode = 90
        stderr = (
            stderr.rstrip()
            + "\nQ-Chem command returned zero but the output has no normal termination marker."
        ).lstrip()
    if returncode == 0:
        write_qchem_provenance(input_path, output_path)
    return QChemJobResult(
        input_path=input_path,
        output_path=output_path,
        returncode=returncode,
        stdout=completed.stdout,
        stderr=stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Q-Chem inputs listed in a file.")
    parser.add_argument("filelist", help="Path to a file listing Q-Chem input files.")
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--command-template", default="qchem {input} {output}")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--skip-failed-jobs", action="store_true")
    args = parser.parse_args(argv)

    results = run_qchem_filelist(
        args.filelist,
        processes=args.processes,
        command_template=args.command_template,
        skip_existing_outputs=not args.rerun_existing,
        skip_failed_jobs=args.skip_failed_jobs,
        verbose=True,
    )
    skipped = sum(result.skipped for result in results)
    print(f"Finished {len(results)} Q-Chem job(s); skipped {skipped} existing output(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
