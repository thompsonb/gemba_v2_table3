# Vendored MT Metrics Eval subset

This directory contains only the MT Metrics Eval modules required for the
WMT24-on-WMT23 PCE and tie-calibrated pairwise-accuracy tasks.

- Upstream: <https://github.com/google-research/mt-metrics-eval>
- Upstream commit: `68a481aea1a787392d55af8f76ac8ef40c5f4664`
- License: Apache License 2.0; see `LICENSE`

Local changes retain the previously added task/PCE progress callbacks. The
unused Apache Beam `parallel_file` implementation and stored human-rating
reader are omitted, allowing the vendored package to depend only on NumPy and
SciPy. Sequential significance tests, including the default 1,000-permutation
protocol, remain available.

## Vendored data

The top-level `wmt23_data/` directory is a curated subset of the MT Metrics Eval V2
bundle downloaded from:

```text
https://data.statmt.org/wmt26/mt-metrics-eval-v2.tgz
```

The original bundle was stored outside the repository under
`~/.mt-metrics-eval/mt-metrics-eval-v2/wmt23/`. Its required files were copied
here, renamed from `wmt23/` to `wmt23_data/`, and the surrounding
`data/mtme-v2/` hierarchy was deliberately removed
so this directory is both the Python-package root and the MTME data root.

Only `en-de`, `he-en`, and `zh-en` resources needed to score WMT23 system
outputs and reproduce Table 3 were retained:

- 3 document maps and 3 source files;
- 6 references, 45 system outputs, and 15 aggregate human-score files; and
- 129 segment-level metric-score files: every source-based or standard-
  reference metric variant admitted by the six WMT24-on-WMT23 tasks.

This is 201 files totaling 86,446,929 bytes (82.44 MiB). Metric system- and
domain-level files were removed because both PCE and tie-calibrated pairwise
accuracy consume segment scores. Nonstandard-reference variants, detailed
human-rating files, other language pairs, and other WMT years were also
removed. The curated files produce the same Table 3 output as the complete
external bundle.

The WMT23 General MT data is released for research use subject to citation and
any source-specific requirements. Cite the WMT23 shared-task overview and the
MT Metrics Eval project when using these files.
