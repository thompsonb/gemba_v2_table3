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
    self.assertIsNone(args.gemba_output_dir)

  def test_metric_list_is_exactly_the_non_v2_table_rows(self):
    self.assertEqual(len(SCRIPT.TABLE3_METRICS), 29)
    self.assertIn("prismRef", SCRIPT.TABLE3_METRICS)
    self.assertNotIn("MS-COMET-QE-22[noref]", SCRIPT.TABLE3_METRICS)
    self.assertFalse(any("gemba-v2" in m.lower() for m in SCRIPT.TABLE3_METRICS))

  def test_local_gemba_metric_name_comes_from_manifest_model(self):
    first = SCRIPT._gemba_metric_name("publisher/First Model")
    second = SCRIPT._gemba_metric_name("publisher/Second Model")
    self.assertNotEqual(first, second)
    for basename, display_name in (first, second):
      self.assertEqual(display_name, f"{basename}[noref]")
      self.assertEqual(basename, basename.lower())
      self.assertNotIn("/", basename)
      self.assertTrue(basename.startswith("gemba-v2-"))
      self.assertTrue(basename.endswith("-rrwa"))

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

  def test_accepts_gemba_output_directory(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      (root / "manifest.json").write_text("{}", encoding="utf-8")
      (root / "judgements").mkdir()
      self.assertEqual(
          SCRIPT._resolve_gemba_output_dir(root), root.resolve()
      )

  def test_rejects_invalid_gemba_output_directory(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      with self.assertRaisesRegex(ValueError, "manifest.json"):
        SCRIPT._resolve_gemba_output_dir(Path(temp_dir))


class GembaScoresTest(unittest.TestCase):

  def _records(self):
    records = {}
    for system_index, system in enumerate(("system", "refA")):
      for segment_index in range(2):
        for judgement_index in range(2):
          key = f"{system}/{segment_index}/{judgement_index}"
          records[key] = {
              "language_pair": "en-de",
              "system": system,
              "global_segment_index": segment_index,
              "judgement_index": judgement_index,
              "score": -float(
                  system_index * 10 + segment_index + judgement_index
              ),
          }
    return records

  def test_aggregates_complete_judgements_in_memory(self):
    eval_set = SimpleNamespace(
        src=["first", "second"],
        sys_outputs={"system": [], "refA": []},
    )
    scores, histogram = SCRIPT._aggregate_gemba_records(
        eval_set,
        "en-de",
        self._records(),
        num_judgements=2,
        aggregate_scores=sum,
    )
    self.assertEqual(scores["system"], [-1.0, -3.0])
    self.assertEqual(scores["refA"], [-21.0, -23.0])
    self.assertEqual(histogram, {2: 4})

  def test_aggregates_available_judgements_for_partial_segment(self):
    records = self._records()
    records.pop("system/0/1")
    eval_set = SimpleNamespace(
        src=["first", "second"],
        sys_outputs={"system": [], "refA": []},
    )
    scores, histogram = SCRIPT._aggregate_gemba_records(
        eval_set,
        "en-de",
        records,
        num_judgements=2,
        aggregate_scores=sum,
    )
    self.assertEqual(scores["system"][0], 0.0)
    self.assertEqual(histogram, {1: 1, 2: 3})

  def test_imputes_entirely_missing_segment_with_two_way_mean(self):
    records = self._records()
    records.pop("system/0/0")
    records.pop("system/0/1")
    eval_set = SimpleNamespace(
        src=["first", "second"],
        sys_outputs={"system": [], "refA": []},
    )
    scores, histogram = SCRIPT._aggregate_gemba_records(
        eval_set,
        "en-de",
        records,
        num_judgements=2,
        aggregate_scores=sum,
    )
    self.assertAlmostEqual(scores["system"][0], -25 / 3)
    self.assertEqual(histogram, {0: 1, 2: 3})


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
                ["metric", "Avg", "sys (SPA)"],
            ],
            output_format=output_format,
            baselines_metainfo=SimpleNamespace(baseline_metrics=set()),
        )
        self.assertIn("0.600", table)
        self.assertNotIn("17", table)


if __name__ == "__main__":
  unittest.main()
