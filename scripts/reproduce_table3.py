#!/usr/bin/env python3
"""Reproduce Table 3's stored WMT23 metrics under the WMT24 protocol.

This script evaluates the metrics distributed in the mt-metrics-eval WMT23
data bundle. By default it excludes every GEMBA-MQM V2 aggregate and
individual-run result, because those scores are not part of that bundle.
It also excludes human-system outputs, matching the population used for the
published Table 3 values.

Locally generated GEMBA-MQM V2 scores can be added with
``--extra-metric-dir OUTPUT/mtme``.

No expected paper values are embedded in the script. The calculations use the
repository's WMT24-on-WMT23 task definition:

* system-level pairwise confidence error (reported as 1-PCE); and
* group-by-item segment-level pairwise accuracy with tie calibration.

Example, run from the parent workspace:

  uv run --frozen python scripts/reproduce_table3.py \
    --permutations 0 \
    --output table3-without-gemba-v2.txt

The default of 1,000 resampling permutations follows the full evaluation
protocol and can be slow. Use ``--permutations 0`` for a point-estimate-only
run; per-task rank annotations are omitted in that mode.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "mt-metrics-eval"
LANGUAGE_PAIRS = ("en-de", "he-en", "zh-en")
GEMBA_V2_METRIC = "GEMBA-MQM-V2[noref]"

# Stored MTME metrics present in Table 3, in the paper's displayed order.
# GEMBA-MQM V2 and its individual runs are intentionally absent.
TABLE3_METRICS = (
    "GEMBA-MQM[noref]",
    "MetricX-23-QE-b[noref]",
    "XCOMET-Ensemble",
    "MetricX-23-QE-c[noref]",
    "XCOMET-XXL",
    "MetricX-23-b",
    "CometKiwi-XXL[noref]",
    "XCOMET-XL",
    "cometoid22-wmt23[noref]",
    "MetricX-23",
    "XCOMET-QE-Ensemble[noref]",
    "CometKiwi-XL[noref]",
    "MetricX-23-QE[noref]",
    "COMET",
    "MetricX-23-c",
    "CometKiwi[noref]",
    "mbr-metricx-qe[noref]",
    "KG-BERTScore[noref]",
    "BLEURT-20",
    "docWMT22CometDA",
    "docWMT22CometKiwiDA[noref]",
    "cometoid22-wmt21[noref]",
    "cometoid22-wmt22[noref]",
    "instructscore",
    "sescoreX",
    "YiSi-1",
    "MaTESe",
    "Calibri-COMET22",
    "prismRef",
)


def _nonnegative_int(value: str) -> int:
  parsed = int(value)
  if parsed < 0:
    raise argparse.ArgumentTypeError("value must be nonnegative")
  return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Reproduce Table 3's stored WMT23 metrics under the WMT24 retrofit "
          "protocol, optionally adding locally generated GEMBA-MQM V2 scores."
      )
  )
  parser.add_argument(
      "--data-dir",
      type=Path,
      default=DEFAULT_DATA_DIR,
      help=(
          "MTME data root containing wmt23/ "
          f"(default: {DEFAULT_DATA_DIR})"
      ),
  )
  parser.add_argument(
      "--permutations",
      type=_nonnegative_int,
      default=1000,
      help=(
          "Resampling draws for metric significance comparisons (default: "
          "1000; use 0 only for a fast point-estimate run)"
      ),
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=0,
      help="NumPy seed for significance resampling (default: 0)",
  )
  parser.add_argument(
      "--extra-metric-dir",
      type=Path,
      help=(
          "Additional MTME root containing wmt23/metric-scores/, such as "
          "the OUTPUT/mtme directory produced by score_wmt23_gemba_v2.py"
      ),
  )
  parser.add_argument(
      "--format",
      choices=("text", "tsv", "latex"),
      default="text",
      help="Output table format (default: text)",
  )
  parser.add_argument(
      "--output",
      type=Path,
      help="Write the table to this file instead of standard output",
  )
  return parser.parse_args(argv)


def _resolve_data_root(path: Path) -> Path:
  root = path.expanduser().resolve()
  if not any((root / name).is_dir() for name in ("wmt23", "wmt23_data")):
    raise ValueError(
        f"MTME data root does not contain wmt23/ or wmt23_data/: {root}"
    )
  return root


def _resolve_extra_metric_root(path: Path | None) -> Path | None:
  if path is None:
    return None
  root = path.expanduser().resolve()
  metric_root = root / "wmt23" / "metric-scores"
  if not metric_root.is_dir():
    raise ValueError(
        f"Extra metric root does not contain wmt23/metric-scores/: {root}"
    )
  return root


def _progress(message: str) -> None:
  """Write an immediately visible progress message without polluting output."""
  print(message, file=sys.stderr, flush=True)


def _import_mtme() -> tuple[Any, Any, Any, Any]:
  try:
    import numpy as np
    from mt_metrics_eval import data
    from mt_metrics_eval import meta_info
    from mt_metrics_eval import tasks
  except ModuleNotFoundError as error:
    raise RuntimeError(
        "Could not import mt-metrics-eval and its dependencies. From the "
        "workspace root, run `uv sync --frozen`, then invoke this script "
        "with `uv run --frozen python scripts/reproduce_table3.py`. "
        f"Original error: {error}"
    ) from error
  return data, meta_info, tasks, np


def _build_eval_sets(
    data_module: Any,
    data_root: Path,
    extra_metric_root: Path | None = None,
) -> dict[tuple[str, str], Any]:
  eval_sets = {}
  for language_pair in LANGUAGE_PAIRS:
    label = f"load WMT23 data and metric scores for {language_pair}"
    _progress(f"[start] {label}")
    start = time.monotonic()
    eval_set = data_module.EvalSet(
        "wmt23",
        language_pair,
        read_stored_metric_scores=True,
        path=(
            [str(data_root), str(extra_metric_root)]
            if extra_metric_root
            else str(data_root)
        ),
    )
    # Evaluate every stored variant. Mark all basenames primary only to avoid
    # adding contrastive-submission markers to the reproduced row labels.
    eval_set.SetPrimaryMetrics(eval_set.metric_basenames)
    eval_sets[("wmt23", language_pair)] = eval_set
    _progress(
        f"[done] {label} ({time.monotonic() - start:.1f}s); "
        f"{len(eval_set.metric_names)} metric variants"
    )
  return eval_sets


def _run_tasks(
    tasks_module: Any,
    numpy_module: Any,
    eval_sets: dict[tuple[str, str], Any],
    permutations: int,
    seed: int,
) -> tuple[Any, list[float]]:
  numpy_module.random.seed(seed)
  task_set, weights = tasks_module.WMT24OnWMT23(
      primary=False, k=permutations
  )
  # Although MTME's helper includes human outputs for he-en, the published
  # Table 3 point estimates were computed without human-system submissions.
  _exclude_human_systems(task_set)
  task_results = []
  _progress(
      f"[evaluate] {len(task_set)} tasks; human systems excluded; "
      f"permutations={permutations}; seed={seed}"
  )
  for index, task in enumerate(task_set, start=1):
    statistic = (
        "acc-t" if task.corr_fcn == "KendallWithTiesOpt" else task.corr_fcn
    )
    label = (
        f"task {index}/{len(task_set)}: {task.lang} "
        f"{task.level} ({statistic})"
    )
    _progress(f"[start] {label}")
    start = time.monotonic()
    result = task.Run(
        eval_sets,
        progress_callback=lambda message: _progress(f"[detail] {message}"),
    )
    task_results.append(result)
    _progress(
        f"[done] {label} ({time.monotonic() - start:.1f}s); "
        f"evaluated {len(result.metrics)} metrics"
    )
  return tasks_module.TaskSetResults(task_results), weights


def _exclude_human_systems(task_set: Any) -> None:
  for task in task_set:
    task.human = False


def _ordinal_scores(scores: dict[str, float]) -> dict[str, tuple[float, int]]:
  return {
      metric: (score, rank)
      for rank, (metric, score) in enumerate(scores.items(), start=1)
  }


def _metric_display_info(
    metric: str, baselines_metainfo: Any
) -> tuple[str, str, bool, bool]:
  """Return basename, no-reference suffix, baseline, and contrastive flags."""
  noref = metric.endswith("[noref]")
  basename = metric[:-len("[noref]")] if noref else metric
  contrastive = basename.startswith("*")
  if contrastive:
    basename = basename[1:]
  baseline = basename in baselines_metainfo.baseline_metrics
  return basename, "[noref]" if noref else "", baseline, contrastive


def _point_estimate_table(
    metrics: Sequence[str],
    average_column: dict[str, tuple[float, int]],
    task_columns: Sequence[dict[str, tuple[float, int]]],
    column_headers: Sequence[Sequence[str]],
    output_format: str,
    baselines_metainfo: Any,
) -> str:
  """Format point estimates while omitting unavailable cluster ranks."""

  def TextMetric(metric: str) -> str:
    basename, noref, baseline, contrastive = _metric_display_info(
        metric, baselines_metainfo
    )
    prefix = "*" if contrastive else "_" if baseline else ""
    return f"{prefix}{basename}{noref}"

  if output_format == "tsv":
    rows = []
    for header_index, header in enumerate(column_headers):
      if header_index == len(column_headers) - 1:
        rows.append(
            [header[0], f"{header[1]} rank", header[1], *header[2:]]
        )
      else:
        rows.append([header[0], header[1], "", *header[2:]])
    for metric in metrics:
      average, rank = average_column[metric]
      rows.append([
          TextMetric(metric),
          str(rank),
          f"{average:f}",
          *(f"{column[metric][0]:f}" for column in task_columns),
      ])
    return "".join("\t".join(row) + "\n" for row in rows)

  if output_format == "latex":
    def Escape(value: str) -> str:
      return value.replace("_", "\\_")

    def LatexMetric(metric: str) -> str:
      basename, noref, baseline, contrastive = _metric_display_info(
          metric, baselines_metainfo
      )
      rendered = Escape(basename) + ("*" if noref else "")
      if contrastive:
        return f"\\textit{{{rendered}}}"
      if baseline:
        return f"\\underline{{{rendered}}}"
      return rendered

    lines = [
        "\\begin{tabular}{l|rr" + "|r" * len(task_columns) + "}",
        "\\toprule",
    ]
    for header in column_headers:
      cells = [Escape(header[0]), f"\\multicolumn{{2}}{{|l}}{{{Escape(header[1])}}}"]
      cells.extend(Escape(value) for value in header[2:])
      lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    for metric in metrics:
      average, rank = average_column[metric]
      cells = [LatexMetric(metric), str(rank), f"{average:.3f}"]
      cells.extend(f"{column[metric][0]:.3f}" for column in task_columns)
      lines.append(" & ".join(cells) + " \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    return "\n".join(lines) + "\n"

  rows = []
  for metric in metrics:
    average, rank = average_column[metric]
    rows.append([
        TextMetric(metric),
        f"{rank}{average:6.3f}",
        *(f"{column[metric][0]:6.3f}" for column in task_columns),
    ])
  widths = [
      max(len(row[index]) for row in [*column_headers, *rows])
      for index in range(len(rows[0]))
  ]
  lines = []
  for header in column_headers:
    lines.append("  ".join(
        value.ljust(widths[index]) if index == 0 else value.rjust(widths[index])
        for index, value in enumerate(header)
    ))
  lines.append("  ".join("-" * width for width in widths))
  for row in rows:
    lines.append("  ".join(
        value.ljust(widths[index]) if index == 0 else value.rjust(widths[index])
        for index, value in enumerate(row)
    ))
  return "\n".join(lines) + "\n"


def _build_table(
    tasks_module: Any,
    meta_info_module: Any,
    results: Any,
    weights: Sequence[float],
    permutations: int,
    output_format: str,
    include_gemba_v2: bool = False,
) -> str:
  average_scores = results.AverageCorrs(weights)
  requested_metrics = list(TABLE3_METRICS)
  if include_gemba_v2:
    requested_metrics.append(GEMBA_V2_METRIC)
  missing = [metric for metric in requested_metrics if metric not in average_scores]
  if missing:
    raise ValueError(
        "MTME data is missing Table 3 metrics: " + ", ".join(missing)
    )
  paper_order = {metric: index for index, metric in enumerate(requested_metrics)}
  metrics = sorted(
      requested_metrics,
      key=lambda metric: (-average_scores[metric], paper_order[metric]),
  )
  average_column = _ordinal_scores({
      metric: average_scores[metric] for metric in metrics
  })

  language_header = ["", ""] + [
      result.attr_vals["lang"] for result in results.results
  ]
  statistic_header = ["metric", "Avg"]
  for result in results.results:
    level = result.attr_vals["level"]
    corr_fcn = result.attr_vals["corr_fcn"]
    statistic = "acc-t" if corr_fcn == "KendallWithTiesOpt" else corr_fcn
    statistic_header.append(f"{level} ({statistic})")

  task_columns = [result.corr_ranks for result in results.results]
  headers = [language_header, statistic_header]
  if permutations == 0:
    return _point_estimate_table(
        metrics,
        average_column,
        task_columns,
        headers,
        output_format,
        meta_info_module.WMT23,
    )
  return tasks_module.MetricsTable(
      metrics=metrics,
      columns=[average_column, *task_columns],
      column_headers=headers,
      fmt=output_format,
      which_metrics="listed",
      baselines_metainfo=meta_info_module.WMT23,
  )


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  try:
    data_root = _resolve_data_root(args.data_dir)
    extra_metric_root = _resolve_extra_metric_root(args.extra_metric_dir)
    _progress(f"[setup] MTME data: {data_root}")
    if extra_metric_root:
      _progress(f"[setup] extra metrics: {extra_metric_root}")
    _progress("[setup] importing mt-metrics-eval")
    data_module, meta_info_module, tasks_module, numpy_module = _import_mtme()
    _progress("[setup] imports complete")
    eval_sets = _build_eval_sets(
        data_module, data_root, extra_metric_root
    )
    results, weights = _run_tasks(
        tasks_module,
        numpy_module,
        eval_sets,
        args.permutations,
        args.seed,
    )
    _progress("[table] formatting results")
    table = _build_table(
        tasks_module,
        meta_info_module,
        results,
        weights,
        args.permutations,
        args.format,
        include_gemba_v2=extra_metric_root is not None,
    )
  except KeyboardInterrupt:
    _progress("[stopped] interrupted by user")
    return 130
  except (OSError, RuntimeError, ValueError) as error:
    _progress(f"error: {error}")
    return 2

  if args.output:
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table, encoding="utf-8")
    _progress(f"[complete] wrote {output_path.resolve()}")
  else:
    print(table, end="" if table.endswith("\n") else "\n")
    _progress("[complete] wrote table to standard output")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
