"""Adversarial tests for the independent conformance harness."""

import ast
import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import spec_oracle as oracle  # noqa: E402
import benchmark_manifest as benchmark  # noqa: E402


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads((ROOT / "tests/spec/v0/cases.json").read_text(encoding="utf-8"))

    def mutant(self):
        return copy.deepcopy(self.document)

    def test_frozen_cases(self):
        cases = oracle.validate_fixture(self.document)
        self.assertGreater(len(cases), 0)
        for case in cases:
            with self.subTest(case=case["id"]):
                self.assertEqual(oracle.execute(case), case["expected"])

    def test_wrong_value(self):
        mutant = self.mutant()
        mutant["cases"][0]["expected"]["value"] = 13
        oracle.validate_fixture(mutant)
        self.assertNotEqual(oracle.execute(mutant["cases"][0]), mutant["cases"][0]["expected"])

    def test_validator_fails_on_mutant_fixture(self):
        mutant = self.mutant()
        mutant["cases"][0]["expected"]["value"] = 13
        with tempfile.TemporaryDirectory(prefix="th003-mutant-") as temporary:
            path = Path(temporary) / "mutant.json"
            path.write_text(json.dumps(mutant), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(oracle.validate(path), 1)
            self.assertIn("FAIL C01-positive", output.getvalue())
            self.assertNotIn("PASS", output.getvalue())

    def test_wrong_shape(self):
        mutant = self.mutant()
        mutant["cases"][4]["expected"]["shape"] = [1, 4]
        oracle.validate_fixture(mutant)
        self.assertNotEqual(oracle.execute(mutant["cases"][4]), mutant["cases"][4]["expected"])

    def test_wrong_dtype(self):
        mutant = self.mutant()
        mutant["cases"][0]["expected"]["dtype"] = "i32"
        with self.assertRaises(oracle.FixtureFailure):
            oracle.validate_fixture(mutant)

    def test_wrong_error_id(self):
        mutant = self.mutant()
        mutant["cases"][2]["expected"]["error_id"] = oracle.ERROR_IDS["bounds"]
        oracle.validate_fixture(mutant)
        self.assertNotEqual(oracle.execute(mutant["cases"][2]), mutant["cases"][2]["expected"])

    def test_overflow_check_mutant(self):
        case = next(case for case in self.document["cases"] if case["id"] == "C02-max")
        with patch.object(oracle, "checked", side_effect=lambda number: number):
            self.assertNotEqual(oracle.execute(case), case["expected"])

    def test_missing_expected(self):
        mutant = self.mutant()
        del mutant["cases"][0]["expected"]
        with self.assertRaises(oracle.FixtureFailure):
            oracle.validate_fixture(mutant)

    def test_unknown_operation(self):
        mutant = self.mutant()
        mutant["cases"][0]["operation"] = "compile_thiran"
        with self.assertRaises(oracle.FixtureFailure):
            oracle.validate_fixture(mutant)

    def test_duplicate_id(self):
        mutant = self.mutant()
        mutant["cases"][1]["id"] = mutant["cases"][0]["id"]
        with self.assertRaises(oracle.FixtureFailure):
            oracle.validate_fixture(mutant)

    def test_zero_cases(self):
        mutant = self.mutant()
        mutant["cases"] = []
        with self.assertRaises(oracle.FixtureFailure):
            oracle.validate_fixture(mutant)

    def test_unknown_selected_case_cannot_pass(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(oracle.FixtureFailure):
            oracle.validate(ROOT / "tests/spec/v0/cases.json", "not-a-case")
        self.assertNotIn("PASS", output.getvalue())

    def test_framework_and_production_import_boundary(self):
        forbidden = {"torch", "numpy", "triton", "tensorflow", "jax", "generated",
                     "src", "ir", "optimizer", "runtime"}
        module_tree = ast.parse((ROOT / "tools/spec_oracle.py").read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(module_tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        self.assertFalse(imports & forbidden)
        oracle.validate_fixture(self.document)
        oracle.execute(self.document["cases"][0])
        self.assertFalse(forbidden & set(sys.modules))


class BenchmarkManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads((ROOT / "tests/spec/benchmark_manifest_example.json").read_text(encoding="utf-8"))

    def test_unmeasured_example_is_structurally_valid(self):
        self.assertEqual(benchmark.validate_document(self.document), 1)

    def test_every_required_field_is_enforced(self):
        for field in benchmark.FIELDS:
            with self.subTest(field=field):
                mutant = copy.deepcopy(self.document)
                del mutant["claims"][0][field]
                with self.assertRaises(benchmark.ManifestFailure):
                    benchmark.validate_document(mutant)

    def test_zero_records_and_unknown_category(self):
        mutant = copy.deepcopy(self.document)
        mutant["claims"] = []
        with self.assertRaises(benchmark.ManifestFailure):
            benchmark.validate_document(mutant)
        mutant = copy.deepcopy(self.document)
        mutant["claims"][0]["category"] = "MODEL_SPEED_FROM_KERNEL"
        with self.assertRaises(benchmark.ManifestFailure):
            benchmark.validate_document(mutant)

    def test_sample_count_mismatch(self):
        mutant = copy.deepcopy(self.document)
        mutant["claims"][0]["raw_samples"] = [1.0]
        mutant["claims"][0]["measured_iteration_count"] = 2
        with self.assertRaises(benchmark.ManifestFailure):
            benchmark.validate_document(mutant)


if __name__ == "__main__":
    unittest.main()
