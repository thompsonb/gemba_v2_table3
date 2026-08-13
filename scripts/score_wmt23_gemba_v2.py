#!/usr/bin/env python3
"""Score WMT23 system outputs with GEMBA-MQM V2 through vLLM replicas.

The scorer follows the paper's segmentation protocol: each request judges one
aligned source/hypothesis segment, while the system prompt contains the full
newline-joined source document. MTME's document maps provide the authoritative
document boundaries.

Successful individual judgments are appended to resumable JSONL files.
Completed segment aggregates are rebuilt from those judgments in memory when
needed. Once every MTME system for a language pair has been scored, its
judgment JSONL is compressed into a verified ZIP archive.
"""

from __future__ import annotations

import argparse
from concurrent import futures
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "mt-metrics-eval"
LANGUAGE_PAIRS = ("en-de", "he-en", "zh-en")
LANGUAGES = {
    "en-de": ("English", "German"),
    "he-en": ("Hebrew", "English"),
    "zh-en": ("Chinese", "English"),
}
DEFAULT_NUM_JUDGEMENTS = 10
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 4096
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ScoreJob:
  language_pair: str
  source_language: str
  target_language: str
  system: str
  document_id: str
  document_index: int
  document_segment_index: int
  global_segment_index: int
  source_document: tuple[str, ...]
  source: str
  target: str

  @property
  def key(self) -> str:
    return _record_key(
        self.language_pair, self.system, self.global_segment_index
    )


def _positive_int(value: str) -> int:
  parsed = int(value)
  if parsed < 1:
    raise argparse.ArgumentTypeError("value must be positive")
  return parsed


def _nonnegative_int(value: str) -> int:
  parsed = int(value)
  if parsed < 0:
    raise argparse.ArgumentTypeError("value must be nonnegative")
  return parsed


def _nonnegative_float(value: str) -> float:
  parsed = float(value)
  if parsed < 0:
    raise argparse.ArgumentTypeError("value must be nonnegative")
  return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Score WMT23 translations with GEMBA-MQM V2 using MTME document "
          "boundaries and one or more local vLLM servers."
      )
  )
  parser.add_argument(
      "--data-dir",
      type=Path,
      default=DEFAULT_DATA_DIR,
      help="MTME data root containing wmt23/",
  )
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=Path("gemba-v2-scores"),
      help="Resumable judgments and manifest (default: gemba-v2-scores)",
  )
  parser.add_argument(
      "--model",
      required=True,
      help="Model name exposed by the local vLLM server",
  )
  parser.add_argument(
      "--model-revision",
      help="Optional checkpoint/revision label recorded in the manifest",
  )
  parser.add_argument(
      "--base-url",
      action="append",
      dest="base_urls",
      help=(
          "vLLM OpenAI-compatible base URL; repeat to use independent "
          "replicas (default: VLLM_BASE_URL or http://127.0.0.1:8000/v1)"
      ),
  )
  parser.add_argument(
      "--cache-salt",
      help=(
          "Optional vLLM prefix-cache namespace; useful for isolated "
          "throughput benchmarks"
      ),
  )
  parser.add_argument(
      "--language-pair",
      action="append",
      choices=LANGUAGE_PAIRS,
      dest="language_pairs",
      help="Language pair to process; repeat as needed (default: all three)",
  )
  parser.add_argument(
      "--system",
      action="append",
      dest="systems",
      help="Exact MTME system name; repeat as needed (default: every system)",
  )
  parser.add_argument(
      "--exclude-human-systems",
      action="store_true",
      help=(
          "Skip reference/human systems. This reduces work for Table 3, but "
          "a complete in-memory MTME metric cannot be constructed until they "
          "are scored."
      ),
  )
  parser.add_argument(
      "--num-judgements",
      type=_positive_int,
      default=DEFAULT_NUM_JUDGEMENTS,
      help=f"Judgments per segment (default: {DEFAULT_NUM_JUDGEMENTS})",
  )
  parser.add_argument(
      "--temperature",
      type=_nonnegative_float,
      default=DEFAULT_TEMPERATURE,
      help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
  )
  parser.add_argument(
      "--max-tokens",
      type=_positive_int,
      default=DEFAULT_MAX_TOKENS,
      help=f"Maximum output tokens per judgment (default: {DEFAULT_MAX_TOKENS})",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=0,
      help="Base sampling seed; each segment gets a stable derived seed",
  )
  parser.add_argument(
      "--max-inflight-judgements",
      type=_positive_int,
      default=128,
      help=(
          "Hard maximum number of simultaneously generating judgments per "
          "vLLM endpoint; every API request generates exactly one judgment "
          "regardless of --num-judgements (default: 128)"
      ),
  )
  parser.add_argument(
      "--retries",
      type=_nonnegative_int,
      default=3,
      help="Retries after the first failed request (default: 3)",
  )
  parser.add_argument(
      "--retry-backoff",
      type=_nonnegative_float,
      default=2.0,
      help="Initial exponential retry delay in seconds (default: 2)",
  )
  parser.add_argument(
      "--progress-every",
      type=_positive_int,
      default=100,
      help="Print progress after this many attempted judgments (default: 100)",
  )
  parser.add_argument(
      "--limit",
      type=_positive_int,
      help="Score at most this many currently unfinished segments",
  )
  parser.add_argument(
      "--max-prompt-characters",
      type=_positive_int,
      help=(
          "Fail before scoring if any rendered message content exceeds this "
          "many characters; vLLM still enforces the model's token limit"
      ),
  )
  parser.add_argument(
      "--context-preflight",
      choices=("none", "documents", "all"),
      default="all",
      help=(
          "Use vLLM's actual tokenizer/chat template to check every prompt "
          "(default) or none; 'documents' is retained as an exhaustive "
          "compatibility alias"
      ),
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Validate alignment and report work/context sizes without scoring",
  )
  return parser.parse_args(argv)


def _progress(message: str) -> None:
  print(message, file=sys.stderr, flush=True)


def _resolve_base_urls(values: Sequence[str] | None) -> tuple[str, ...]:
  urls = tuple(
      url.strip().rstrip("/")
      for url in (values or (
          os.environ.get("VLLM_BASE_URL") or DEFAULT_BASE_URL,
      ))
  )
  if len(set(urls)) != len(urls):
    raise ValueError("Every --base-url must identify a distinct endpoint")
  for url in urls:
    parsed = urlparse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
      raise ValueError(f"Invalid vLLM base URL: {url!r}")
  return urls


def _import_dependencies() -> tuple[Any, Any, Any, Any, Any]:
  try:
    from gemba import GembaV2
    from gemba import prompt as gemba_prompt
    from gemba import scoring as gemba_scoring
    from mt_metrics_eval import data as mtme_data
  except ModuleNotFoundError as error:
    raise RuntimeError(
        "Missing local dependencies. Run `uv sync --frozen` from the "
        f"workspace root. Original error: {error}"
    ) from error
  return (
      GembaV2,
      gemba_prompt.build_messages,
      gemba_prompt,
      gemba_scoring,
      mtme_data,
  )


def _resolve_data_root(path: Path) -> Path:
  root = path.expanduser().resolve()
  if not any((root / name).is_dir() for name in ("wmt23", "wmt23_data")):
    raise ValueError(
        f"MTME data root does not contain wmt23/ or wmt23_data/: {root}"
    )
  return root


def _wmt23_data_dir(data_root: Path) -> Path:
  standard = data_root / "wmt23"
  return standard if standard.is_dir() else data_root / "wmt23_data"


def _load_eval_sets(
    mtme_data: Any, data_root: Path, language_pairs: Sequence[str]
) -> dict[str, Any]:
  eval_sets = {}
  for language_pair in language_pairs:
    eval_set = mtme_data.EvalSet(
        "wmt23",
        language_pair,
        read_stored_metric_scores=False,
        path=str(data_root),
    )
    _validate_eval_set(language_pair, eval_set)
    eval_sets[language_pair] = eval_set
    _progress(
        f"[data] {language_pair}: {len(eval_set.docs)} documents, "
        f"{len(eval_set.src)} segments, {len(eval_set.sys_outputs)} systems"
    )
  return eval_sets


def _validate_eval_set(language_pair: str, eval_set: Any) -> None:
  cursor = 0
  for document_id, (start, end) in eval_set.docs.items():
    if start != cursor or not start < end <= len(eval_set.src):
      raise ValueError(
          f"Non-contiguous document map for {language_pair}/{document_id}: "
          f"expected start {cursor}, got [{start}, {end})"
      )
    cursor = end
  if cursor != len(eval_set.src):
    raise ValueError(
        f"Document map for {language_pair} covers {cursor}/{len(eval_set.src)} "
        "segments"
    )
  for system, outputs in eval_set.sys_outputs.items():
    if len(outputs) != len(eval_set.src):
      raise ValueError(
          f"Unaligned system output {language_pair}/{system}: "
          f"{len(outputs)} vs {len(eval_set.src)} segments"
      )


def _select_systems(
    language_pair: str,
    eval_set: Any,
    requested_systems: Sequence[str] | None,
    exclude_human_systems: bool,
) -> list[str]:
  available = set(eval_set.sys_outputs)
  if requested_systems:
    missing = sorted(set(requested_systems) - available)
    if missing:
      raise ValueError(
          f"Unknown {language_pair} system(s): {', '.join(missing)}"
      )
    return sorted(set(requested_systems))
  if exclude_human_systems:
    available -= set(eval_set.human_sys_names)
  return sorted(available)


def _build_jobs(
    eval_sets: dict[str, Any],
    requested_systems: Sequence[str] | None = None,
    exclude_human_systems: bool = False,
) -> list[ScoreJob]:
  jobs = []
  for language_pair, eval_set in eval_sets.items():
    source_language, target_language = LANGUAGES[language_pair]
    systems = _select_systems(
        language_pair,
        eval_set,
        requested_systems,
        exclude_human_systems,
    )
    for system in systems:
      outputs = eval_set.sys_outputs[system]
      for document_index, (document_id, (start, end)) in enumerate(
          eval_set.docs.items()
      ):
        source_document = tuple(eval_set.src[start:end])
        for global_index in range(start, end):
          jobs.append(ScoreJob(
              language_pair=language_pair,
              source_language=source_language,
              target_language=target_language,
              system=system,
              document_id=document_id,
              document_index=document_index,
              document_segment_index=global_index - start,
              global_segment_index=global_index,
              source_document=source_document,
              source=eval_set.src[global_index],
              target=outputs[global_index],
          ))
  return jobs


def _record_key(language_pair: str, system: str, segment_index: int) -> str:
  return f"{language_pair}\t{system}\t{segment_index}"


def _document_key(job: ScoreJob) -> tuple[str, str]:
  return job.language_pair, job.document_id


def _assign_document_endpoints(
    jobs: Sequence[ScoreJob], endpoint_count: int
) -> tuple[dict[tuple[str, str], int], list[int]]:
  """Balance documents across endpoints without splitting any document."""
  if endpoint_count < 1:
    raise ValueError("At least one vLLM endpoint is required")
  weights: dict[tuple[str, str], int] = {}
  for job in jobs:
    key = _document_key(job)
    # Target length approximates annotation/echo decoding cost; source length
    # accounts for the small non-cached, segment-specific user payload.
    weights[key] = weights.get(key, 0) + len(job.source) + len(job.target) + 1

  assignments = {}
  endpoint_loads = [0] * endpoint_count
  for key, weight in sorted(
      weights.items(), key=lambda item: (-item[1], item[0])
  ):
    endpoint = min(
        range(endpoint_count),
        key=lambda index: (endpoint_loads[index], index),
    )
    assignments[key] = endpoint
    endpoint_loads[endpoint] += weight
  return assignments, endpoint_loads


def _stable_seed(
    segment_key: str, base_seed: int, judgement_index: int
) -> int:
  identity = json.dumps(
      [segment_key, base_seed, judgement_index],
      ensure_ascii=False,
      separators=(",", ":"),
  )
  digest = hashlib.sha256(identity.encode("utf-8")).digest()
  return int.from_bytes(digest[:4], "big") % (2**31)


def _prompt_character_count(
    job: ScoreJob, build_messages: Callable[..., list[dict[str, str]]]
) -> int:
  messages = build_messages(
      source_language=job.source_language,
      target_language=job.target_language,
      source_segments=job.source_document,
      source=job.source,
      target=job.target,
  )
  return sum(len(message["content"]) for message in messages)


def _scan_prompt_sizes(
    jobs: Sequence[ScoreJob],
    build_messages: Callable[..., list[dict[str, str]]],
) -> tuple[int, ScoreJob | None]:
  maximum = 0
  maximum_job = None
  for job in jobs:
    characters = _prompt_character_count(job, build_messages)
    if characters > maximum:
      maximum = characters
      maximum_job = job
  return maximum, maximum_job


def _preflight_candidates(
    jobs: Sequence[ScoreJob],
    mode: str,
    build_messages: Callable[..., list[dict[str, str]]],
) -> list[ScoreJob]:
  if mode == "none":
    return []
  if mode in ("all", "documents"):
    # Token count is not monotonic with character or UTF-8 byte length, so no
    # single rendition can safely represent every prompt in a document.
    return list(jobs)
  raise ValueError(f"Unknown context preflight mode: {mode}")


def _messages_for_job(
    job: ScoreJob, build_messages: Callable[..., list[dict[str, str]]]
) -> list[dict[str, str]]:
  return build_messages(
      source_language=job.source_language,
      target_language=job.target_language,
      source_segments=job.source_document,
      source=job.source,
      target=job.target,
  )


def _tokenize_url(base_url: str) -> str:
  parsed = urlparse.urlsplit(base_url)
  path = parsed.path.rstrip("/")
  if path.endswith("/v1"):
    path = path[:-len("/v1")]
  return urlparse.urlunsplit((
      parsed.scheme,
      parsed.netloc,
      f"{path}/tokenize",
      "",
      "",
  ))


def _tokenize_chat(
    base_url: str,
    model: str,
    messages: Sequence[dict[str, str]],
    timeout: float = 60.0,
) -> tuple[int, int]:
  payload = json.dumps({
      "model": model,
      "messages": list(messages),
      "add_generation_prompt": True,
  }).encode("utf-8")
  request = urlrequest.Request(
      _tokenize_url(base_url),
      data=payload,
      headers={
          "Authorization": "Bearer EMPTY",
          "Content-Type": "application/json",
      },
      method="POST",
  )
  try:
    with urlrequest.urlopen(request, timeout=timeout) as response:
      result = json.loads(response.read())
  except (urlerror.URLError, json.JSONDecodeError) as error:
    raise RuntimeError(
        f"vLLM tokenizer preflight failed at {_tokenize_url(base_url)}: "
        f"{error}; use --context-preflight none only if the server lacks "
        "the /tokenize chat endpoint"
    ) from error
  count = result.get("count")
  max_model_len = result.get("max_model_len")
  if not isinstance(count, int) or not isinstance(max_model_len, int):
    raise RuntimeError(
        "vLLM /tokenize response lacks integer count/max_model_len fields"
    )
  return count, max_model_len


def _run_context_preflight(
    jobs: Sequence[ScoreJob],
    mode: str,
    build_messages: Callable[..., list[dict[str, str]]],
    tokenizer: Callable[
        [ScoreJob, Sequence[dict[str, str]]], tuple[int, int]
    ],
    max_tokens: int,
    workers: int,
    progress_every: int,
) -> tuple[int, ScoreJob | None, int | None]:
  candidates = _preflight_candidates(jobs, mode, build_messages)
  if not candidates:
    return 0, None, None
  _progress(
      f"[context] token preflight: {len(candidates)} {mode} candidate(s)"
  )
  maximum = 0
  maximum_job = None
  model_limit = None
  with futures.ThreadPoolExecutor(max_workers=workers) as executor:
    pending: dict[futures.Future[Any], ScoreJob] = {}
    candidate_iter = iter(candidates)

    def submit_next() -> bool:
      try:
        job = next(candidate_iter)
      except StopIteration:
        return False
      future = executor.submit(
          tokenizer, job, _messages_for_job(job, build_messages)
      )
      pending[future] = job
      return True

    for _ in range(min(len(candidates), workers * 2)):
      submit_next()
    completed = 0
    while pending:
      done, _ = futures.wait(
          pending, return_when=futures.FIRST_COMPLETED
      )
      for future in done:
        job = pending.pop(future)
        count, max_model_len = future.result()
        completed += 1
        if model_limit is None:
          model_limit = max_model_len
        elif model_limit != max_model_len:
          raise RuntimeError(
              "vLLM /tokenize returned inconsistent max_model_len values"
          )
        if count + max_tokens > max_model_len:
          raise ValueError(
              f"Prompt {job.key} ({job.document_id}) needs {count} input + "
              f"{max_tokens} output tokens, exceeding model limit "
              f"{max_model_len}; full-document context cannot be preserved"
          )
        if count > maximum:
          maximum = count
          maximum_job = job
        if completed % progress_every == 0 or completed == len(candidates):
          _progress(
              f"[context] tokenized {completed}/{len(candidates)} candidates"
          )
        submit_next()
  return maximum, maximum_job, model_limit


def _sha256_file(path: Path, digest: Any) -> None:
  digest.update(path.name.encode("utf-8"))
  with path.open("rb") as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(chunk)


def _dataset_fingerprint(data_root: Path) -> str:
  digest = hashlib.sha256()
  dataset = _wmt23_data_dir(data_root)
  paths = []
  for language_pair in LANGUAGE_PAIRS:
    paths.extend((
        dataset / "sources" / f"{language_pair}.txt",
        dataset / "documents" / f"{language_pair}.docs",
    ))
    paths.extend(sorted(
        (dataset / "system-outputs" / language_pair).glob("*.*")
    ))
  for path in paths:
    relative = path.relative_to(dataset)
    digest.update(str(relative).encode("utf-8"))
    _sha256_file(path, digest)
  return digest.hexdigest()


def _prompt_fingerprint(gemba_prompt: Any, gemba_scoring: Any) -> str:
  payload = {
      "instructions": gemba_prompt.INSTRUCTIONS,
      "annotation_schema": gemba_prompt.ANNOTATION_SCHEMA,
      "severity_weights": gemba_scoring.SEVERITY_WEIGHTS,
      "minor_punctuation_weight": gemba_scoring.MINOR_PUNCTUATION_WEIGHT,
  }
  serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
  return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _package_fingerprint(module: Any) -> str:
  root = Path(module.__file__).resolve().parent
  digest = hashlib.sha256()
  for path in sorted(root.glob("*.py")):
    digest.update(path.name.encode("utf-8"))
    _sha256_file(path, digest)
  return digest.hexdigest()


def _configuration(
    args: argparse.Namespace,
    data_root: Path,
    gemba_prompt: Any,
    gemba_scoring: Any,
) -> dict[str, Any]:
  try:
    gemba_version = importlib.metadata.version("gemba")
  except importlib.metadata.PackageNotFoundError:
    gemba_version = "unknown"
  return {
      "schema_version": SCHEMA_VERSION,
      "test_set": "wmt23",
      "data_root": str(data_root),
      "dataset_sha256": _dataset_fingerprint(data_root),
      "gemba_version": gemba_version,
      "gemba_code_sha256": _package_fingerprint(gemba_prompt),
      "prompt_sha256": _prompt_fingerprint(gemba_prompt, gemba_scoring),
      "model": args.model,
      "model_revision": args.model_revision,
      "cache_salt": args.cache_salt,
      "num_judgements": args.num_judgements,
      "temperature": args.temperature,
      "max_tokens": args.max_tokens,
      "base_seed": args.seed,
  }


def _ensure_manifest(output_dir: Path, configuration: dict[str, Any]) -> Path:
  output_dir.mkdir(parents=True, exist_ok=True)
  manifest_path = output_dir / "manifest.json"
  if manifest_path.exists():
    try:
      manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
      raise ValueError(f"Invalid JSON in {manifest_path}") from error
    recorded_configuration = manifest.get("configuration")
    comparable_recorded = dict(recorded_configuration or {})
    comparable_current = dict(configuration)
    # Paths and endpoints are runtime infrastructure, not scoring parameters.
    # Dataset identity is enforced by dataset_sha256, so relocating the same
    # curated files must not prevent a resume. Each judgment retains the URL
    # that handled it for auditing.
    comparable_recorded.pop("data_root", None)
    comparable_current.pop("data_root", None)
    comparable_recorded.pop("base_url", None)
    comparable_recorded.pop("base_urls", None)
    comparable_current.pop("base_url", None)
    comparable_current.pop("base_urls", None)
    comparable_recorded.setdefault("cache_salt", None)
    comparable_current.setdefault("cache_salt", None)
    comparable_recorded.pop("max_tokens", None)
    comparable_current.pop("max_tokens", None)
    if comparable_recorded != comparable_current:
      raise ValueError(
          f"Scoring configuration does not match {manifest_path}; use a "
          "different --output-dir for a different model or configuration"
      )
    return manifest_path
  manifest = {
      "created_at": datetime.now(timezone.utc).isoformat(),
      "configuration": configuration,
  }
  manifest_path.write_text(
      json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  return manifest_path


def _judgement_path(output_dir: Path, language_pair: str) -> Path:
  return output_dir / "judgements" / f"{language_pair}.jsonl"


def _judgement_archive_path(output_dir: Path, language_pair: str) -> Path:
  return output_dir / "judgements" / f"{language_pair}.jsonl.zip"


def _judgement_key(segment_key: str, judgement_index: int) -> str:
  return f"{segment_key}\t{judgement_index}"


def _read_records(
    file: Any, source: str
) -> dict[str, dict[str, Any]]:
  records = {}
  for line_number, line in enumerate(file, 1):
    if not line.strip():
      continue
    try:
      record = json.loads(line)
    except json.JSONDecodeError as error:
      raise ValueError(f"Invalid JSON at {source}:{line_number}") from error
    if not isinstance(record, dict):
      raise ValueError(f"Record is not a JSON object at {source}:{line_number}")
    key = record.get("key")
    if not isinstance(key, str):
      raise ValueError(f"Missing record key at {source}:{line_number}")
    if key in records:
      raise ValueError(f"Duplicate record key {key!r} in {source}")
    records[key] = record
  return records


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
  if not path.exists():
    return {}
  records = {}
  truncated_at = None
  truncated_line = None
  with path.open("rb") as file:
    line_number = 0
    while True:
      line_start = file.tell()
      line = file.readline()
      if not line:
        break
      line_number += 1
      if not line.strip():
        continue
      try:
        record = json.loads(line.decode("utf-8"))
      except (UnicodeDecodeError, json.JSONDecodeError) as error:
        # A process interruption can leave only the final append incomplete.
        # Remove that tail so the next append starts at a valid JSONL boundary.
        if not line.endswith(b"\n") and not file.read(1):
          truncated_at = line_start
          truncated_line = line_number
          break
        raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
      if not isinstance(record, dict):
        raise ValueError(
            f"Record is not a JSON object at {path}:{line_number}"
        )
      key = record.get("key")
      if not isinstance(key, str):
        raise ValueError(f"Missing record key at {path}:{line_number}")
      if key in records:
        raise ValueError(f"Duplicate record key {key!r} in {path}")
      records[key] = record
  if truncated_at is not None:
    with path.open("r+b") as file:
      file.truncate(truncated_at)
    _progress(
        f"[recover] removed truncated final record at "
        f"{path}:{truncated_line}"
    )
  return records


def _load_judgements(
    output_dir: Path, language_pair: str
) -> dict[str, dict[str, Any]]:
  path = _judgement_path(output_dir, language_pair)
  archive_path = _judgement_archive_path(output_dir, language_pair)
  records = {}
  if archive_path.exists():
    try:
      with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != [path.name]:
          raise ValueError(
              f"Expected only {path.name!r} in {archive_path}"
          )
        with archive.open(path.name) as binary_file:
          with io.TextIOWrapper(binary_file, encoding="utf-8") as file:
            records = _read_records(
                file, f"{archive_path}!{path.name}"
            )
    except zipfile.BadZipFile as error:
      raise ValueError(f"Invalid ZIP archive: {archive_path}") from error

  loose_records = _load_records(path)
  for key, record in loose_records.items():
    if key in records and records[key] != record:
      raise ValueError(
          f"Conflicting judgment record {key!r} in {path} and {archive_path}"
      )
    records[key] = record
  return records


def _archive_judgements(output_dir: Path, language_pair: str) -> Path:
  path = _judgement_path(output_dir, language_pair)
  archive_path = _judgement_archive_path(output_dir, language_pair)
  if not path.exists():
    if archive_path.exists():
      return archive_path
    raise ValueError(f"Missing completed judgments for {language_pair}: {path}")

  temporary = archive_path.with_name(
      f".{archive_path.name}.{os.getpid()}.tmp"
  )
  try:
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
      archive.write(path, arcname=path.name)
    with zipfile.ZipFile(temporary) as archive:
      failed_member = archive.testzip()
      if failed_member is not None:
        raise ValueError(
            f"CRC verification failed for {failed_member!r} in {temporary}"
        )
    os.replace(temporary, archive_path)
    original_size = path.stat().st_size
    archived_size = archive_path.stat().st_size
    path.unlink()
  except BaseException:
    temporary.unlink(missing_ok=True)
    raise

  _progress(
      f"[archive] {language_pair}: {original_size / 2**20:.1f} MiB -> "
      f"{archived_size / 2**20:.1f} MiB at {archive_path}"
  )
  return archive_path


class _ThreadScorers:

  def __init__(self, factory: Callable[[], Any]):
    self._factory = factory
    self._local = threading.local()

  def get(self) -> Any:
    if not hasattr(self._local, "scorer"):
      self._local.scorer = self._factory()
    return self._local.scorer


def _score_judgement(
    job: ScoreJob,
    judgement_index: int,
    scorers: _ThreadScorers,
    base_url: str,
    base_seed: int,
    retries: int,
    retry_backoff: float,
) -> dict[str, Any]:
  seed = _stable_seed(
      job.key,
      base_seed=base_seed,
      judgement_index=judgement_index,
  )
  for attempt in range(retries + 1):
    try:
      result = scorers.get().score_segment(
          source_segments=job.source_document,
          source=job.source,
          target=job.target,
          source_language=job.source_language,
          target_language=job.target_language,
          seed=seed,
      )
      result_values = result.to_dict()
      run_scores = result_values.get("run_scores")
      annotations = result_values.get("annotations")
      if not isinstance(run_scores, (list, tuple)) or len(run_scores) != 1:
        raise RuntimeError(
            "Single-judgment scorer returned an unexpected number of scores"
        )
      if not isinstance(annotations, (list, tuple)) or len(annotations) != 1:
        raise RuntimeError(
            "Single-judgment scorer returned an unexpected number of annotations"
        )
      return {
          "key": _judgement_key(job.key, judgement_index),
          "segment_key": job.key,
          "judgement_index": judgement_index,
          "language_pair": job.language_pair,
          "system": job.system,
          "document_id": job.document_id,
          "global_segment_index": job.global_segment_index,
          "seed": seed,
          "vllm_base_url": base_url,
          "completed_at": datetime.now(timezone.utc).isoformat(),
          "score": float(run_scores[0]),
          "annotation": annotations[0],
      }
    except Exception as error:  # The final error is recorded by the main thread.
      if attempt == retries:
        raise RuntimeError(
            f"{job.key} judgement {judgement_index + 1} failed after "
            f"{retries + 1} attempt(s): {error}"
        ) from error
      time.sleep(retry_backoff * (2**attempt))
  raise AssertionError("unreachable")


def _validate_judgement_record(
    record: dict[str, Any], segment_key: str, judgement_index: int
) -> None:
  key = _judgement_key(segment_key, judgement_index)
  if record.get("key") != key:
    raise ValueError(f"Judgement record has the wrong key for {key!r}")
  if record.get("segment_key") != segment_key:
    raise ValueError(f"Judgement record has the wrong segment key for {key!r}")
  if record.get("judgement_index") != judgement_index:
    raise ValueError(f"Judgement record has the wrong index for {key!r}")
  score = record.get("score")
  if not isinstance(score, (int, float)) or not math.isfinite(score):
    raise ValueError(f"Judgement record {key!r} has an invalid score")
  if not isinstance(record.get("annotation"), Mapping):
    raise ValueError(f"Judgement record {key!r} has an invalid annotation")


def _aggregate_job(
    job: ScoreJob,
    judgement_records: Mapping[int, dict[str, Any]],
    num_judgements: int,
    aggregate_scores: Callable[[Sequence[float]], float],
) -> dict[str, Any]:
  if set(judgement_records) != set(range(num_judgements)):
    raise ValueError(f"Cannot aggregate incomplete judgements for {job.key}")
  ordered = []
  for index in range(num_judgements):
    record = judgement_records[index]
    _validate_judgement_record(record, job.key, index)
    ordered.append(record)
  run_scores = [float(record["score"]) for record in ordered]
  return {
      "key": job.key,
      "score": float(aggregate_scores(run_scores)),
      "run_scores": run_scores,
  }


def _aggregate_completed_jobs(
    jobs: Sequence[ScoreJob],
    judgements: Mapping[str, dict[str, Any]],
    num_judgements: int,
    aggregate_scores: Callable[[Sequence[float]], float],
) -> dict[str, dict[str, Any]]:
  records = {}
  for job in jobs:
    job_judgements = {}
    for judgement_index in range(num_judgements):
      key = _judgement_key(job.key, judgement_index)
      if key not in judgements:
        continue
      record = judgements[key]
      _validate_judgement_record(record, job.key, judgement_index)
      job_judgements[judgement_index] = record
    if len(job_judgements) == num_judgements:
      records[job.key] = _aggregate_job(
          job,
          job_judgements,
          num_judgements,
          aggregate_scores,
      )
  return records


def _write_json_line(file: Any, record: dict[str, Any]) -> None:
  file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
  file.write("\n")
  file.flush()


def _run_jobs(
    jobs: Sequence[ScoreJob],
    scorer_factories: Sequence[Callable[[], Any]],
    base_urls: Sequence[str],
    document_endpoints: Mapping[tuple[str, str], int],
    records: dict[str, dict[str, Any]],
    judgements: dict[str, dict[str, Any]],
    output_dir: Path,
    num_judgements: int,
    max_inflight_judgements_per_endpoint: int,
    base_seed: int,
    retries: int,
    retry_backoff: float,
    progress_every: int,
    aggregate_scores: Callable[[Sequence[float]], float],
) -> int:
  if not jobs:
    return 0
  (output_dir / "judgements").mkdir(parents=True, exist_ok=True)
  language_pairs = sorted({job.language_pair for job in jobs})
  judgement_files = {
      language_pair: _judgement_path(output_dir, language_pair).open(
          "a", encoding="utf-8"
      )
      for language_pair in language_pairs
  }
  if not scorer_factories:
    raise ValueError("At least one scorer factory is required")
  if len(base_urls) != len(scorer_factories):
    raise ValueError("Every scorer factory must have one base URL")
  scorers = [_ThreadScorers(factory) for factory in scorer_factories]
  endpoint_jobs = [[] for _ in scorer_factories]
  for job in jobs:
    endpoint = document_endpoints[_document_key(job)]
    if not 0 <= endpoint < len(scorer_factories):
      raise ValueError(f"Invalid endpoint {endpoint} for {job.key}")
    endpoint_jobs[endpoint].append(job)
  # Keep requests sharing a full-document prefix close together as well as
  # pinned to one endpoint, making reuse more likely before cache eviction.
  for endpoint_group in endpoint_jobs:
    endpoint_group.sort(key=lambda job: (
        job.language_pair,
        job.document_index,
        job.system,
        job.global_segment_index,
    ))

  job_judgements: dict[str, dict[int, dict[str, Any]]] = {
      job.key: {} for job in jobs
  }
  for job in jobs:
    for judgement_index in range(num_judgements):
      key = _judgement_key(job.key, judgement_index)
      if key not in judgements:
        continue
      record = judgements[key]
      _validate_judgement_record(record, job.key, judgement_index)
      job_judgements[job.key][judgement_index] = record

  segment_succeeded = 0

  def finish_segment_if_ready(job: ScoreJob) -> bool:
    nonlocal segment_succeeded
    if job.key in records:
      return False
    if len(job_judgements[job.key]) != num_judgements:
      return False
    record = _aggregate_job(
        job,
        job_judgements[job.key],
        num_judgements,
        aggregate_scores,
    )
    records[job.key] = record
    segment_succeeded += 1
    return True

  for job in jobs:
    finish_segment_if_ready(job)

  endpoint_tasks: list[list[tuple[ScoreJob, int]]] = [
      [] for _ in scorer_factories
  ]
  for endpoint, endpoint_group in enumerate(endpoint_jobs):
    for job in endpoint_group:
      if job.key in records:
        continue
      for judgement_index in range(num_judgements):
        key = _judgement_key(job.key, judgement_index)
        if key not in judgements:
          endpoint_tasks[endpoint].append((job, judgement_index))

  total_judgements = sum(map(len, endpoint_tasks))
  attempted = judgement_succeeded = failed = 0
  start = time.monotonic()
  executors = [
      futures.ThreadPoolExecutor(
          max_workers=max_inflight_judgements_per_endpoint
      )
      for _ in scorer_factories
  ]
  pending: dict[futures.Future[Any], tuple[ScoreJob, int, int]] = {}
  task_iters = [iter(endpoint) for endpoint in endpoint_tasks]

  def submit_next(endpoint: int) -> bool:
    try:
      job, judgement_index = next(task_iters[endpoint])
    except StopIteration:
      return False
    future = executors[endpoint].submit(
        _score_judgement,
        job,
        judgement_index,
        scorers[endpoint],
        base_urls[endpoint],
        base_seed,
        retries,
        retry_backoff,
    )
    pending[future] = (job, judgement_index, endpoint)
    return True

  try:
    for endpoint, endpoint_group in enumerate(endpoint_tasks):
      for _ in range(
          min(
              len(endpoint_group),
              max_inflight_judgements_per_endpoint * 2,
          )
      ):
        submit_next(endpoint)
    while pending:
      done, _ = futures.wait(
          pending, return_when=futures.FIRST_COMPLETED
      )
      for future in done:
        job, judgement_index, endpoint = pending.pop(future)
        attempted += 1
        try:
          judgement = future.result()
        except Exception as error:
          failed += 1
          _progress(f"[error] {error}")
        else:
          judgement_succeeded += 1
          judgements[judgement["key"]] = judgement
          job_judgements[job.key][judgement_index] = judgement
          _write_json_line(
              judgement_files[job.language_pair], judgement
          )
          finish_segment_if_ready(job)
        submit_next(endpoint)
        if attempted % progress_every == 0 or attempted == total_judgements:
          elapsed = time.monotonic() - start
          rate = attempted / elapsed if elapsed else 0.0
          _progress(
              f"[progress] judgments {attempted}/{total_judgements} "
              f"attempted; {judgement_succeeded} succeeded; {failed} "
              f"failed; {segment_succeeded}/{len(jobs)} segments completed; "
              f"{rate:.2f} judgments/s"
          )
  except KeyboardInterrupt:
    for future in pending:
      future.cancel()
    for executor in executors:
      executor.shutdown(wait=False, cancel_futures=True)
    raise
  else:
    for executor in executors:
      executor.shutdown(wait=True)
  finally:
    for file in judgement_files.values():
      file.close()
  return failed


def _language_pair_complete(
    language_pair: str,
    eval_set: Any,
    records: Mapping[str, dict[str, Any]],
) -> bool:
  required_keys = [
      _record_key(language_pair, system, segment_index)
      for system in sorted(eval_set.sys_outputs)
      for segment_index in range(len(eval_set.src))
  ]
  missing = [key for key in required_keys if key not in records]
  if missing:
    _progress(
        f"[status] {language_pair}: incomplete "
        f"({len(required_keys) - len(missing)}/{len(required_keys)} segments)"
    )
    return False
  _progress(f"[status] {language_pair}: complete ({len(required_keys)} segments)")
  return True


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  try:
    (
        GembaV2,
        build_messages,
        gemba_prompt,
        gemba_scoring,
        mtme_data,
    ) = _import_dependencies()
    data_root = _resolve_data_root(args.data_dir)
    output_dir = args.output_dir.expanduser().resolve()
    language_pairs = tuple(dict.fromkeys(
        args.language_pairs or LANGUAGE_PAIRS
    ))
    eval_sets = _load_eval_sets(mtme_data, data_root, language_pairs)
    jobs = _build_jobs(
        eval_sets,
        requested_systems=args.systems,
        exclude_human_systems=args.exclude_human_systems,
    )
    base_urls = _resolve_base_urls(args.base_urls)
    document_endpoints, endpoint_loads = _assign_document_endpoints(
        jobs, len(base_urls)
    )
    maximum_chars, maximum_job = _scan_prompt_sizes(jobs, build_messages)
    if maximum_job:
      _progress(
          f"[context] maximum rendered message content: {maximum_chars} "
          f"characters at {maximum_job.key} ({maximum_job.document_id})"
      )
    if args.max_prompt_characters and maximum_chars > args.max_prompt_characters:
      raise ValueError(
          f"Maximum prompt size {maximum_chars} exceeds "
          f"--max-prompt-characters {args.max_prompt_characters}"
      )
    selected_counts = {
        language_pair: sum(
            job.language_pair == language_pair for job in jobs
        )
        for language_pair in language_pairs
    }
    _progress(
        "[plan] selected segments: "
        + ", ".join(
            f"{language_pair}={selected_counts[language_pair]}"
            for language_pair in language_pairs
        )
        + f"; total={len(jobs)}; judgments={len(jobs) * args.num_judgements}"
    )
    for endpoint, base_url in enumerate(base_urls):
      document_count = sum(
          assigned == endpoint for assigned in document_endpoints.values()
      )
      segment_count = sum(
          document_endpoints[_document_key(job)] == endpoint for job in jobs
      )
      _progress(
          f"[routing] endpoint {endpoint + 1}/{len(base_urls)} {base_url}: "
          f"{document_count} complete documents, {segment_count} segments, "
          f"{segment_count * args.num_judgements} judgments, "
          f"estimated work={endpoint_loads[endpoint]}"
      )
    if args.dry_run:
      _progress("[complete] dry run; no files written and no model requests sent")
      return 0

    configuration = _configuration(
        args, data_root, gemba_prompt, gemba_scoring
    )
    if args.context_preflight != "none":
      maximum_tokens, maximum_token_job, model_limit = (
          _run_context_preflight(
              jobs,
              args.context_preflight,
              build_messages,
              lambda job, messages: _tokenize_chat(
                  base_urls[document_endpoints[_document_key(job)]],
                  args.model,
                  messages,
              ),
              args.max_tokens,
              args.max_inflight_judgements * len(base_urls),
              args.progress_every,
          )
      )
      if maximum_token_job:
        _progress(
            f"[context] maximum checked prompt: {maximum_tokens} input "
            f"tokens at {maximum_token_job.key} "
            f"({maximum_token_job.document_id}); model limit={model_limit}; "
            f"reserved output={args.max_tokens}"
        )
    manifest_path = _ensure_manifest(output_dir, configuration)
    _progress(f"[setup] manifest: {manifest_path}")
    judgements = {}
    for language_pair in language_pairs:
      judgements.update(_load_judgements(output_dir, language_pair))
    records = _aggregate_completed_jobs(
        jobs,
        judgements,
        args.num_judgements,
        gemba_scoring.aggregate_scores,
    )
    _progress(f"[resume] loaded {len(judgements)} saved judgments")
    _progress(f"[resume] rebuilt {len(records)} completed segment aggregates")

    pending_jobs = [job for job in jobs if job.key not in records]
    if args.limit is not None:
      pending_jobs = pending_jobs[:args.limit]
    pending_judgements = sum(
        _judgement_key(job.key, judgement_index) not in judgements
        for job in pending_jobs
        for judgement_index in range(args.num_judgements)
    )
    _progress(
        f"[start] {len(pending_jobs)} pending segments; "
        f"{pending_judgements} pending judgments; "
        f"endpoints={len(base_urls)}; n={args.num_judgements}; "
        f"max-inflight-judgements="
        f"{args.max_inflight_judgements}/endpoint; "
        "one judgment/request; "
        f"maximum concurrency="
        f"{args.max_inflight_judgements * len(base_urls)} judgments; "
        f"retries={args.retries}"
    )
    scorer_factories = [
        lambda base_url=base_url: GembaV2(
            model=args.model,
            base_url=base_url,
            num_judgements=1,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            cache_salt=args.cache_salt,
        )
        for base_url in base_urls
    ]

    failures = _run_jobs(
        pending_jobs,
        scorer_factories,
        base_urls,
        document_endpoints,
        records,
        judgements,
        output_dir,
        args.num_judgements,
        args.max_inflight_judgements,
        args.seed,
        args.retries,
        args.retry_backoff,
        args.progress_every,
        gemba_scoring.aggregate_scores,
    )

    completions = [
        _language_pair_complete(
            language_pair,
            eval_sets[language_pair],
            records,
        )
        for language_pair in language_pairs
    ]
    for language_pair, complete in zip(language_pairs, completions):
      if complete:
        _archive_judgements(output_dir, language_pair)
    if failures:
      _progress(
          f"[incomplete] {failures} judgments failed; rerun the same command "
          "to retry only unfinished judgments"
      )
      return 1
    _progress("[complete] scoring pass finished")
    return 0
  except KeyboardInterrupt:
    _progress("[stopped] interrupted; saved judgments are resumable")
    return 130
  except (OSError, RuntimeError, ValueError) as error:
    _progress(f"error: {error}")
    return 2


if __name__ == "__main__":
  raise SystemExit(main())
