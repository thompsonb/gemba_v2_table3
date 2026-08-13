#!/usr/bin/env python3
"""Unit tests for the resumable WMT23 GEMBA-MQM V2 scorer."""

from __future__ import annotations

from collections import OrderedDict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).with_name("score_wmt23_gemba_v2.py")
SPEC = importlib.util.spec_from_file_location("score_wmt23_gemba_v2", SCRIPT_PATH)
SCRIPT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SCRIPT
SPEC.loader.exec_module(SCRIPT)


def _fake_eval_set():
  return SimpleNamespace(
      src=["source zero", "source one", "source two"],
      docs=OrderedDict((
          ("document-a", [0, 2]),
          ("document-b", [2, 3]),
      )),
      sys_outputs={
          "system": ["target zero", "target one", "target two"],
          "refA": ["reference zero", "reference one", "reference two"],
      },
      human_sys_names={"refA"},
  )


class ArgumentsTest(unittest.TestCase):

  def test_defaults_cover_all_systems_and_paper_parameters(self):
    args = SCRIPT._parse_args(["--model", "model"])
    self.assertFalse(args.exclude_human_systems)
    self.assertEqual(args.num_judgements, 10)
    self.assertEqual(args.temperature, 0.4)
    self.assertEqual(args.max_tokens, 4096)
    self.assertEqual(args.max_inflight_judgements, 128)
    self.assertEqual(args.seed, 0)
    self.assertEqual(args.context_preflight, "all")
    self.assertIsNone(args.base_urls)
    self.assertEqual(args.data_dir, SCRIPT.DEFAULT_DATA_DIR)
    self.assertTrue((args.data_dir / "wmt23_data").is_dir())
    self.assertNotIn("export_only", vars(args))

  def test_base_url_can_be_repeated(self):
    args = SCRIPT._parse_args([
        "--model", "model",
        "--base-url", "http://127.0.0.1:8000/v1",
        "--base-url", "http://127.0.0.1:8001/v1",
    ])
    self.assertEqual(args.base_urls, [
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.1:8001/v1",
    ])

  def test_base_urls_are_normalized_and_must_be_distinct(self):
    self.assertEqual(
        SCRIPT._resolve_base_urls([" http://server-a/v1/ "]),
        ("http://server-a/v1",),
    )
    with self.assertRaisesRegex(ValueError, "distinct endpoint"):
      SCRIPT._resolve_base_urls([
          "http://server-a/v1", "http://server-a/v1/",
      ])

class SegmentationTest(unittest.TestCase):

  def test_jobs_preserve_complete_document_context(self):
    eval_set = _fake_eval_set()
    SCRIPT._validate_eval_set("en-de", eval_set)
    jobs = SCRIPT._build_jobs({"en-de": eval_set})
    self.assertEqual(len(jobs), 6)
    first = next(job for job in jobs if job.system == "system")
    self.assertEqual(first.source_document, ("source zero", "source one"))
    self.assertEqual(first.source, "source zero")
    self.assertEqual(first.target, "target zero")
    last = [job for job in jobs if job.system == "system"][-1]
    self.assertEqual(last.source_document, ("source two",))
    self.assertEqual(last.document_segment_index, 0)
    self.assertEqual(last.global_segment_index, 2)

  def test_human_systems_can_be_excluded(self):
    jobs = SCRIPT._build_jobs(
        {"en-de": _fake_eval_set()}, exclude_human_systems=True
    )
    self.assertEqual({job.system for job in jobs}, {"system"})
    self.assertEqual(len(jobs), 3)

  def test_document_gaps_are_rejected(self):
    eval_set = _fake_eval_set()
    eval_set.docs["document-b"] = [1, 3]
    with self.assertRaisesRegex(ValueError, "Non-contiguous"):
      SCRIPT._validate_eval_set("en-de", eval_set)

  def test_stable_seed_depends_on_complete_judgement_identity(self):
    first = SCRIPT._stable_seed(
        "en-de\tsystem\t0", base_seed=0, judgement_index=0
    )
    self.assertEqual(
        first,
        SCRIPT._stable_seed(
            "en-de\tsystem\t0", base_seed=0, judgement_index=0
        ),
    )
    self.assertNotEqual(
        first,
        SCRIPT._stable_seed(
            "en-de\tsystem\t1", base_seed=0, judgement_index=0
        ),
    )
    self.assertNotEqual(
        first,
        SCRIPT._stable_seed(
            "en-de\tsystem\t0", base_seed=0, judgement_index=1
        ),
    )

  def test_base_seeds_select_disjoint_judgement_seeds(self):
    segment_key = "en-de\tAIRC\t42"
    first_run = {
        SCRIPT._stable_seed(
            segment_key, base_seed=0, judgement_index=index
        )
        for index in range(10)
    }
    second_run = {
        SCRIPT._stable_seed(
            segment_key, base_seed=1, judgement_index=index
        )
        for index in range(10)
    }
    self.assertEqual(len(first_run), 10)
    self.assertEqual(len(second_run), 10)
    self.assertTrue(first_run.isdisjoint(second_run))

  def test_document_routing_never_splits_a_document(self):
    jobs = SCRIPT._build_jobs({"en-de": _fake_eval_set()})
    assignments, loads = SCRIPT._assign_document_endpoints(jobs, 2)
    self.assertEqual(set(assignments), {
        ("en-de", "document-a"),
        ("en-de", "document-b"),
    })
    self.assertEqual(set(assignments.values()), {0, 1})
    self.assertTrue(all(load > 0 for load in loads))
    for job in jobs:
      self.assertEqual(
          assignments[(job.language_pair, job.document_id)],
          assignments[SCRIPT._document_key(job)],
      )

  def test_legacy_document_preflight_alias_is_exhaustive(self):
    jobs = SCRIPT._build_jobs({"en-de": _fake_eval_set()})

    def BuildMessages(**values):
      return [
          {"role": "system", "content": "\n".join(values["source_segments"])},
          {"role": "user", "content": values["target"]},
      ]

    candidates = SCRIPT._preflight_candidates(
        jobs, "documents", BuildMessages
    )
    self.assertEqual(candidates, jobs)

  def test_context_preflight_reserves_output_tokens(self):
    jobs = SCRIPT._build_jobs(
        {"en-de": _fake_eval_set()}, exclude_human_systems=True
    )[:1]

    def BuildMessages(**values):
      del values
      return [{"role": "user", "content": "prompt"}]

    with self.assertRaisesRegex(ValueError, "exceeding model limit"):
      SCRIPT._run_context_preflight(
          jobs,
          "all",
          BuildMessages,
          tokenizer=lambda job, messages: (90, 100),
          max_tokens=11,
          workers=1,
          progress_every=1,
      )

  def test_vllm_tokenizer_url_is_outside_v1(self):
    self.assertEqual(
        SCRIPT._tokenize_url("http://127.0.0.1:8000/v1"),
        "http://127.0.0.1:8000/tokenize",
    )


class GembaApiTest(unittest.TestCase):

  def test_score_segment_uses_full_context_and_request_seed(self):
    from gemba import GembaV2

    annotation = {
        "source_language": "English",
        "source": "source zero",
        "target_language": "German",
        "target": "target zero",
        "errors": {"critical": [], "major": [], "minor": []},
    }

    class Completions:

      def __init__(self):
        self.request = None

      def create(self, **request):
        self.request = request
        choice = SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content=json.dumps(annotation)),
        )
        return SimpleNamespace(choices=[choice, choice])

    completions = Completions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    scorer = GembaV2(
        model="model",
        num_judgements=2,
        cache_salt="test-cache-namespace",
        client=client,
    )
    result = scorer.score_segment(
        source_segments=["source zero", "source one"],
        source="source zero",
        target="target zero",
        source_language="English",
        target_language="German",
        seed=123,
    )
    self.assertEqual(result.score, 0.0)
    self.assertEqual(completions.request["seed"], 123)
    self.assertEqual(
        completions.request["extra_body"],
        {"cache_salt": "test-cache-namespace"},
    )
    system_prompt = completions.request["messages"][0]["content"]
    self.assertIn("source zero\nsource one", system_prompt)
    user_payload = json.loads(completions.request["messages"][1]["content"])
    self.assertEqual(user_payload["source"], "source zero")
    self.assertEqual(user_payload["target"], "target zero")


class DispatchTest(unittest.TestCase):

  def test_run_jobs_persists_individual_judgements_and_routes_documents(self):
    jobs = SCRIPT._build_jobs({"en-de": _fake_eval_set()})
    assignments, _ = SCRIPT._assign_document_endpoints(jobs, 2)
    base_urls = ("http://server-a/v1", "http://server-b/v1")

    class Result:

      def to_dict(self):
        return {
            "score": 0.0,
            "run_scores": [0.0],
            "filtered_run_scores": [0.0],
            "annotations": [{"errors": {}}],
        }

    class Scorer:

      def score_segment(self, **unused):
        return Result()

    records = {}
    judgements = {}
    with tempfile.TemporaryDirectory() as temp_dir:
      output_dir = Path(temp_dir)
      failures = SCRIPT._run_jobs(
          jobs,
          (Scorer, Scorer),
          base_urls,
          assignments,
          records,
          judgements,
          output_dir,
          num_judgements=2,
          max_inflight_judgements_per_endpoint=1,
          base_seed=0,
          retries=0,
          retry_backoff=0,
          progress_every=len(jobs),
          aggregate_scores=lambda scores: sum(scores) / len(scores),
      )
      persisted = SCRIPT._load_records(
          output_dir / "judgements" / "en-de.jsonl"
      )
      self.assertFalse((output_dir / "raw").exists())
      self.assertFalse((output_dir / "errors").exists())
    self.assertEqual(failures, 0)
    self.assertEqual(len(records), len(jobs))
    self.assertEqual(len(judgements), len(jobs) * 2)
    self.assertEqual(persisted, judgements)
    rebuilt = SCRIPT._aggregate_completed_jobs(
        jobs,
        judgements,
        num_judgements=2,
        aggregate_scores=lambda scores: sum(scores) / len(scores),
    )
    self.assertEqual(rebuilt, records)
    document_urls = {}
    for judgement in judgements.values():
      key = (judgement["language_pair"], judgement["document_id"])
      document_urls.setdefault(key, set()).add(judgement["vllm_base_url"])
      self.assertEqual(
          judgement["vllm_base_url"], base_urls[assignments[key]]
      )
    self.assertTrue(all(len(urls) == 1 for urls in document_urls.values()))

  def test_only_failed_judgement_is_retried(self):
    job = SCRIPT._build_jobs(
        {"en-de": _fake_eval_set()}, exclude_human_systems=True
    )[0]
    assignments = {SCRIPT._document_key(job): 0}
    judgement_seeds = [
        SCRIPT._stable_seed(
            job.key, base_seed=0, judgement_index=index
        )
        for index in range(3)
    ]

    class Result:

      def __init__(self, score):
        self.score = score

      def to_dict(self):
        return {
            "score": self.score,
            "run_scores": [self.score],
            "filtered_run_scores": [self.score],
            "annotations": [{"score": self.score}],
        }

    class FlakyScorer:

      def __init__(self):
        self.calls = []

      def score_segment(self, seed, **unused):
        self.calls.append(seed)
        if seed == judgement_seeds[1] and self.calls.count(seed) == 1:
          raise RuntimeError("one transient failure")
        return Result(float(seed))

    scorer = FlakyScorer()
    records = {}
    judgements = {}
    with tempfile.TemporaryDirectory() as temp_dir:
      failures = SCRIPT._run_jobs(
          [job],
          (lambda: scorer,),
          ("http://server/v1",),
          assignments,
          records,
          judgements,
          Path(temp_dir),
          num_judgements=3,
          max_inflight_judgements_per_endpoint=1,
          base_seed=0,
          retries=1,
          retry_backoff=0,
          progress_every=3,
          aggregate_scores=lambda scores: sum(scores) / len(scores),
      )
    self.assertEqual(failures, 0)
    self.assertEqual(len(records), 1)
    self.assertEqual(len(judgements), 3)
    self.assertEqual(scorer.calls, [
        judgement_seeds[0],
        judgement_seeds[1],
        judgement_seeds[1],
        judgement_seeds[2],
    ])

  def test_resume_submits_only_missing_judgement(self):
    job = SCRIPT._build_jobs(
        {"en-de": _fake_eval_set()}, exclude_human_systems=True
    )[0]
    assignments = {SCRIPT._document_key(job): 0}
    judgement_seeds = [
        SCRIPT._stable_seed(
            job.key, base_seed=0, judgement_index=index
        )
        for index in range(3)
    ]

    class Result:

      def to_dict(self):
        return {
            "score": 0.0,
            "run_scores": [0.0],
            "filtered_run_scores": [0.0],
            "annotations": [{"errors": {}}],
        }

    class Scorer:

      def __init__(self, fail_seed=None):
        self.fail_seed = fail_seed
        self.calls = []

      def score_segment(self, seed, **unused):
        self.calls.append(seed)
        if seed == self.fail_seed:
          raise RuntimeError("terminal failure")
        return Result()

    records = {}
    judgements = {}
    first = Scorer(fail_seed=judgement_seeds[1])
    second = Scorer()
    common = {
        "jobs": [job],
        "base_urls": ("http://server/v1",),
        "document_endpoints": assignments,
        "records": records,
        "judgements": judgements,
        "num_judgements": 3,
        "max_inflight_judgements_per_endpoint": 1,
        "base_seed": 0,
        "retries": 0,
        "retry_backoff": 0,
        "progress_every": 3,
        "aggregate_scores": lambda scores: sum(scores) / len(scores),
    }
    with tempfile.TemporaryDirectory() as temp_dir:
      common["output_dir"] = Path(temp_dir)
      failures = SCRIPT._run_jobs(
          scorer_factories=(lambda: first,), **common
      )
      self.assertEqual(failures, 1)
      self.assertEqual(len(records), 0)
      self.assertEqual(len(judgements), 2)
      self.assertFalse((Path(temp_dir) / "errors").exists())
      failures = SCRIPT._run_jobs(
          scorer_factories=(lambda: second,), **common
      )
    self.assertEqual(failures, 0)
    self.assertEqual(len(records), 1)
    self.assertEqual(len(judgements), 3)
    self.assertEqual(second.calls, [judgement_seeds[1]])


class PersistenceTest(unittest.TestCase):

  def test_truncated_final_record_is_removed_before_resume(self):
    valid_record = {"key": "complete", "score": 0.0}
    valid_line = json.dumps(valid_record).encode("utf-8") + b"\n"
    with tempfile.TemporaryDirectory() as temp_dir:
      path = Path(temp_dir) / "judgements.jsonl"
      path.write_bytes(valid_line + b'{"key":"partial"')
      self.assertEqual(
          SCRIPT._load_records(path), {"complete": valid_record}
      )
      self.assertEqual(path.read_bytes(), valid_line)
      resumed_record = {"key": "resumed", "score": -1.0}
      with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(resumed_record) + "\n")
      self.assertEqual(SCRIPT._load_records(path), {
          "complete": valid_record,
          "resumed": resumed_record,
      })

  def test_malformed_complete_final_record_is_rejected(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      path = Path(temp_dir) / "judgements.jsonl"
      path.write_bytes(b'{"key":"broken"\n')
      with self.assertRaisesRegex(ValueError, "Invalid JSON"):
        SCRIPT._load_records(path)

  def test_non_object_record_is_rejected(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      path = Path(temp_dir) / "judgements.jsonl"
      path.write_text("[]\n", encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "not a JSON object"):
        SCRIPT._load_records(path)

  def test_duplicate_judgement_record_is_rejected(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      path = Path(temp_dir) / "judgements.jsonl"
      line = json.dumps({"key": "same", "score": 0}) + "\n"
      path.write_text(line + line, encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "Duplicate record key"):
        SCRIPT._load_records(path)

  def test_completed_judgements_are_archived_and_loaded(self):
    records = {
        "first": {"key": "first", "score": 0.0},
        "second": {"key": "second", "score": -1.0},
    }
    lines = "".join(
        json.dumps(record) + "\n" for record in records.values()
    )
    with tempfile.TemporaryDirectory() as temp_dir:
      output_dir = Path(temp_dir)
      path = SCRIPT._judgement_path(output_dir, "en-de")
      path.parent.mkdir(parents=True)
      path.write_text(lines, encoding="utf-8")
      archive_path = SCRIPT._archive_judgements(output_dir, "en-de")
      self.assertFalse(path.exists())
      self.assertTrue(archive_path.exists())
      self.assertEqual(
          SCRIPT._load_judgements(output_dir, "en-de"), records
      )

  def test_archive_loader_tolerates_identical_loose_copy(self):
    record = {"key": "same", "score": 0.0}
    line = json.dumps(record) + "\n"
    with tempfile.TemporaryDirectory() as temp_dir:
      output_dir = Path(temp_dir)
      path = SCRIPT._judgement_path(output_dir, "en-de")
      path.parent.mkdir(parents=True)
      path.write_text(line, encoding="utf-8")
      SCRIPT._archive_judgements(output_dir, "en-de")
      # This represents interruption after the atomic archive installation but
      # before removal of the source JSONL.
      path.write_text(line, encoding="utf-8")
      self.assertEqual(
          SCRIPT._load_judgements(output_dir, "en-de"),
          {"same": record},
      )

  def test_manifest_rejects_configuration_change(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      output_dir = Path(temp_dir)
      SCRIPT._ensure_manifest(output_dir, {"model": "first"})
      with self.assertRaisesRegex(ValueError, "does not match"):
        SCRIPT._ensure_manifest(output_dir, {"model": "second"})

  def test_manifest_allows_max_tokens_change(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      output_dir = Path(temp_dir)
      SCRIPT._ensure_manifest(
          output_dir, {"model": "same", "max_tokens": 2048}
      )
      SCRIPT._ensure_manifest(
          output_dir, {"model": "same", "max_tokens": 4096}
      )

  def test_manifest_allows_endpoint_change(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      output_dir = Path(temp_dir)
      SCRIPT._ensure_manifest(output_dir, {
          "model": "same",
          "base_url": "http://server-a/v1",
      })
      SCRIPT._ensure_manifest(output_dir, {
          "model": "same",
          "base_urls": ["http://server-a/v1", "http://server-b/v1"],
      })

  def test_manifest_allows_identical_dataset_to_move(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      output_dir = Path(temp_dir)
      SCRIPT._ensure_manifest(output_dir, {
          "model": "same",
          "data_root": "/old/cache/mtme-v2",
          "dataset_sha256": "same-dataset",
      })
      SCRIPT._ensure_manifest(output_dir, {
          "model": "same",
          "data_root": "/project/mt-metrics-eval",
          "dataset_sha256": "same-dataset",
      })

  def test_complete_language_pair_is_detected(self):
    eval_set = _fake_eval_set()
    records = {}
    for system in sorted(eval_set.sys_outputs):
      for segment_index in range(len(eval_set.src)):
        key = SCRIPT._record_key("en-de", system, segment_index)
        records[key] = {"key": key}
    self.assertTrue(
        SCRIPT._language_pair_complete("en-de", eval_set, records)
    )

  def test_incomplete_language_pair_is_detected(self):
    self.assertFalse(
        SCRIPT._language_pair_complete("en-de", _fake_eval_set(), {})
    )


if __name__ == "__main__":
  unittest.main()
