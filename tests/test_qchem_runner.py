from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from UMRR.qchem_runner import (
    QChemJobError,
    qchem_output_matches_input,
    run_qchem_filelist,
    write_qchem_provenance,
)


class QChemRunnerTests(unittest.TestCase):
    def test_skips_existing_outputs_with_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scan_dir = Path(tmpdir)
            input_path = scan_dir / "mode_0_v0.0.in"
            output_path = scan_dir / "mode_0_v0.0.out"
            filelist_path = scan_dir / "filelist.txt"
            input_path.write_text("$molecule\n$end\n", encoding="utf-8")
            output_path.write_text(
                "Total energy =  -1.0\nThank you very much for using Q-Chem\n",
                encoding="utf-8",
            )
            filelist_path.write_text("mode_0_v0.0\n", encoding="utf-8")
            write_qchem_provenance(input_path, output_path)

            results = run_qchem_filelist(
                filelist_path,
                processes=1,
                command_template="this-command-should-not-run {input} {output}",
                skip_existing_outputs=True,
            )

            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].skipped)
            self.assertTrue(results[0].input_path.is_absolute())
            self.assertEqual(results[0].output_path, output_path.resolve())

    def test_changed_input_forces_existing_output_to_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scan_dir = Path(tmpdir)
            input_path = scan_dir / "point.in"
            output_path = scan_dir / "point.out"
            script_path = scan_dir / "fake_qchem.py"
            filelist_path = scan_dir / "filelist.txt"
            input_path.write_text("old method\n", encoding="utf-8")
            output_path.write_text(
                "Total energy = -1.0\nThank you very much for using Q-Chem\n",
                encoding="utf-8",
            )
            write_qchem_provenance(input_path, output_path)
            input_path.write_text("new method\n", encoding="utf-8")
            filelist_path.write_text("point.in\n", encoding="utf-8")
            script_path.write_text(
                "import pathlib, sys\n"
                "pathlib.Path(sys.argv[2]).write_text("
                "'Total energy = -2.0\\nThank you very much for using Q-Chem\\n')\n",
                encoding="utf-8",
            )

            results = run_qchem_filelist(
                filelist_path,
                processes=1,
                command_template=f"{sys.executable} {script_path} {{input}} {{output}}",
                skip_existing_outputs=True,
            )

            self.assertFalse(results[0].skipped)
            self.assertIn("-2.0", output_path.read_text(encoding="utf-8"))
            self.assertTrue(qchem_output_matches_input(input_path, output_path))

    def test_zero_return_without_normal_termination_is_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scan_dir = Path(tmpdir)
            input_path = scan_dir / "point.in"
            script_path = scan_dir / "fake_qchem.py"
            filelist_path = scan_dir / "filelist.txt"
            input_path.write_text("$molecule\n$end\n", encoding="utf-8")
            filelist_path.write_text("point.in\n", encoding="utf-8")
            script_path.write_text(
                "import pathlib, sys\npathlib.Path(sys.argv[2]).write_text('partial output\\n')\n",
                encoding="utf-8",
            )

            with self.assertRaises(QChemJobError) as context:
                run_qchem_filelist(
                    filelist_path,
                    processes=1,
                    command_template=f"{sys.executable} {script_path} {{input}} {{output}}",
                    skip_existing_outputs=False,
                )

            self.assertIn("no normal termination marker", str(context.exception))

    def test_can_continue_after_failed_return_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scan_dir = Path(tmpdir)
            input_path = scan_dir / "point.in"
            output_path = scan_dir / "point.out"
            script_path = scan_dir / "fake_qchem.py"
            filelist_path = scan_dir / "filelist.txt"
            input_path.write_text("$molecule\n$end\n", encoding="utf-8")
            filelist_path.write_text("point.in\n", encoding="utf-8")
            script_path.write_text(
                "import pathlib, sys\n"
                "pathlib.Path(sys.argv[2]).write_text('Total energy = -1.0\\n')\n"
                "sys.stderr.write(':q!\\n')\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )

            results = run_qchem_filelist(
                filelist_path,
                processes=1,
                command_template=f"{sys.executable} {script_path} {{input}} {{output}}",
                skip_existing_outputs=False,
                skip_failed_jobs=True,
            )

            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].failed)
            self.assertTrue(output_path.exists())

    def test_strict_failure_message_includes_output_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scan_dir = Path(tmpdir)
            input_path = scan_dir / "point.in"
            script_path = scan_dir / "fake_qchem.py"
            filelist_path = scan_dir / "filelist.txt"
            input_path.write_text("$molecule\n$end\n", encoding="utf-8")
            filelist_path.write_text("point.in\n", encoding="utf-8")
            script_path.write_text(
                "import pathlib, sys\n"
                "pathlib.Path(sys.argv[2]).write_text('partial output\\n')\n"
                "sys.stderr.write('bad geometry\\n')\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )

            with self.assertRaises(QChemJobError) as context:
                run_qchem_filelist(
                    filelist_path,
                    processes=1,
                    command_template=f"{sys.executable} {script_path} {{input}} {{output}}",
                    skip_existing_outputs=False,
                    skip_failed_jobs=False,
                )

            message = str(context.exception)
            self.assertIn("Q-Chem job failed", message)
            self.assertIn("bad geometry", message)
            self.assertIn("partial output", message)


if __name__ == "__main__":
    unittest.main()
