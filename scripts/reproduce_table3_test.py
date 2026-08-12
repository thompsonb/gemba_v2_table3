#!/usr/bin/env python3
"""Unit tests for the Table 3 reproduction script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).with_name("reproduce_table3.py")
SPEC = importlib.util.spec_from_file_location("reproduce_table3", SCRIPT_PATH)
SCRIPT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SCRIPT)


class ArgumentsTest(unittest.TestCase):

  def test_defaults_do_not_require_gemba_scores(self):
    args = SCRIPT._parse_args([])
    self.assertEqual(args.permutations, 1000)
    self.assertEqual(args.data_dir, SCRIPT.DEFAULT_DATA_DIR)
    self.assertTrue((args.data_dir / "wmt23_data").is_dir())
    self.assertNotIn("rank_cutoff", vars(args))
    self.assertNotIn("gemba_scores_dir", vars(args))

  def test_metric_list_is_exactly_the_non_v2_table_rows(self):
    self.assertEqual(len(SCRIPT.TABLE3_METRICS), 29)
    self.assertIn("prismRef", SCRIPT.TABLE3_METRICS)
    self.assertNotIn("MS-COMET-QE-22[noref]", SCRIPT.TABLE3_METRICS)
    self.assertFalse(any("gemba-v2" in m.lower() for m in SCRIPT.TABLE3_METRICS))

  def test_rejects_negative_permutations(self):
    with self.assertRaises(SystemExit):
      SCRIPT._parse_args(["--permutations", "-1"])


class DataRootTest(unittest.TestCase):

  def test_accepts_root_containing_wmt23(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      (root / "wmt23").mkdir()
      self.assertEqual(SCRIPT._resolve_data_root(root), root.resolve())

  def test_accepts_root_containing_flattened_wmt23_data(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      (root / "wmt23_data").mkdir()
      self.assertEqual(SCRIPT._resolve_data_root(root), root.resolve())

  def test_rejects_root_without_wmt23(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      with self.assertRaisesRegex(ValueError, "does not contain wmt23"):
        SCRIPT._resolve_data_root(Path(temp_dir))

  def test_accepts_extra_metric_root(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      (root / "wmt23" / "metric-scores").mkdir(parents=True)
      self.assertEqual(
          SCRIPT._resolve_extra_metric_root(root), root.resolve()
      )

  def test_rejects_invalid_extra_metric_root(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      with self.assertRaisesRegex(ValueError, "does not contain"):
        SCRIPT._resolve_extra_metric_root(Path(temp_dir))


class VendoredDataTest(unittest.TestCase):

  def test_curated_wmt23_file_set_and_size(self):
    root = SCRIPT.DEFAULT_DATA_DIR / "wmt23_data"
    files = [path for path in root.rglob("*") if path.is_file()]
    self.assertEqual(len(files), 201)
    self.assertEqual(sum(path.stat().st_size for path in files), 86_446_929)
    metric_root = root / "metric-scores"
    for language_pair in SCRIPT.LANGUAGE_PAIRS:
      metric_files = list((metric_root / language_pair).glob("*.score"))
      self.assertEqual(len(metric_files), 43)
      self.assertTrue(all(
          path.name.endswith(".seg.score") for path in metric_files
      ))


class RankingTest(unittest.TestCase):

  def test_ordinal_scores_preserve_score_order(self):
    self.assertEqual(
        SCRIPT._ordinal_scores({"a": 0.9, "b": 0.8}),
        {"a": (0.9, 1), "b": (0.8, 2)},
    )


class ProtocolTest(unittest.TestCase):

  def test_human_systems_are_excluded_from_every_task(self):
    task_set = [SimpleNamespace(human=True), SimpleNamespace(human=False)]
    SCRIPT._exclude_human_systems(task_set)
    self.assertTrue(all(task.human is False for task in task_set))

  def test_point_estimate_formats_omit_task_rank(self):
    for output_format in ("text", "tsv", "latex"):
      with self.subTest(output_format=output_format):
        table = SCRIPT._point_estimate_table(
            metrics=["metric"],
            average_column={"metric": (0.5, 1)},
            task_columns=[{"metric": (0.6, 17)}],
            column_headers=[
                ["", "", "en-de"],
                ["metric", "Avg", "sys (pce)"],
            ],
            output_format=output_format,
            baselines_metainfo=SimpleNamespace(baseline_metrics=set()),
        )
        self.assertIn("0.600", table)
        self.assertNotIn("17", table)


if __name__ == "__main__":
  unittest.main()
