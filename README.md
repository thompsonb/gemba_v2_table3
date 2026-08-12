# WMT23 Table 3 reproduction and GEMBA-MQM V2 scoring

This project reproduces the stored WMT23 metric results under the WMT24
retrofit protocol and can score the same translations with GEMBA-MQM V2. The
base table intentionally excludes GEMBA-MQM V2 because those scores are not in
the original WMT23 bundle; locally generated scores can be added afterward.
Human-system outputs are excluded from evaluation to match the paper's
population.

## Setup

Create/update the uv environment from this directory:

```bash
uv sync --frozen
```

The uv project pins Python 3.12 via `.python-version`, installs the minimal
vendored `mt-metrics-eval/` package, and installs GEMBA from the exact private
Git commit recorded in `pyproject.toml` and `uv.lock`. SSH access to that GEMBA
repository is therefore required on first setup.

The required 82.44 MiB WMT23 subset is committed with the vendored MTME code:

```text
mt-metrics-eval/wmt23_data/
```

No separate MTME download is required. The retained 43 task-eligible metric
variants per language pair preserve both point estimates and the optional full
significance-test population. See `mt-metrics-eval/VENDORED.md` for provenance,
the exact pruning rules, and contents.

## Run

From the project root:

```bash
uv run --frozen python scripts/reproduce_table3.py \
  --permutations 0 \
  --output table3-without-gemba-v2.txt
```

To use a non-default MTME data location, add `--data-dir PATH`. The documented
command computes all PCE, acc-t, and average values while skipping the slow
significance-cluster calculation. Per-task rank annotations are omitted in
this mode because ordinary ranks are not the paper's significance clusters.
The average column retains a rank recalculated over the 29 displayed metrics.
Omit `--permutations 0` to use the script's default of 1,000 significance-test
permutations when cluster ranks among the stored metrics are needed.

The script prints flushed progress messages to standard error before and after
loading each language pair and evaluating each of the six tasks. During a PCE
task, it also reports each metric point estimate. When significance testing is
enabled, it additionally reports each metric-pair test.

To select another output format, use `--format text`, `--format tsv`, or
`--format latex`.

## Score WMT23 with GEMBA-MQM V2

The batch scorer uses MTME's original document boundaries. Each request judges
one aligned source/hypothesis segment independently while retaining the entire
newline-joined source document in the system prompt, as specified in the
GEMBA-MQM V2 paper. It is reference-free: reference translations are scored as
candidate systems when producing complete MTME files but are never placed in
the prompt as references.

Start one independent vLLM server on each 72 GB GPU. This avoids cross-GPU NCCL
initialization while allowing the scorer to use both replicas for any language
pair. Each server uses text-only mode, vLLM's generation defaults, prefix
caching for repeated document context, and non-thinking Qwen3.5 output.
DeepGEMM is disabled because its FP8 MoE weight-layout conversion fails for
this checkpoint on these Blackwell cards; vLLM falls back to CUTLASS:

```bash
nohup env \
  CUDA_VISIBLE_DEVICES=1 \
  VLLM_USE_DEEP_GEMM=0 \
  uvx --from vllm vllm serve \
  Qwen/Qwen3.5-35B-A3B-FP8 \
  --port 8000 \
  --generation-config vllm \
  --max-model-len 32768 \
  --language-model-only \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  > vllm-qwen35-gpu1.log 2>&1 &
echo $! > vllm-qwen35-gpu1.pid

nohup env \
  CUDA_VISIBLE_DEVICES=2 \
  VLLM_USE_DEEP_GEMM=0 \
  uvx --from vllm vllm serve \
  Qwen/Qwen3.5-35B-A3B-FP8 \
  --port 8001 \
  --generation-config vllm \
  --max-model-len 32768 \
  --language-model-only \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  > vllm-qwen35-gpu2.log 2>&1 &
echo $! > vllm-qwen35-gpu2.pid
```

Follow both servers until they report that the APIs are listening:

```bash
tail -f vllm-qwen35-gpu1.log vllm-qwen35-gpu2.log
```

Press Ctrl-C to stop following the log; this does not stop the server. Before
starting a scoring command, verify that the API health endpoint returns HTTP
status 200:

```bash
curl --fail --show-error http://127.0.0.1:8000/health
curl --fail --show-error http://127.0.0.1:8001/health
```

A successful health check prints no response body and exits with status zero.
If either command cannot connect, that server is still initializing or has
failed; inspect its log and do not start the scorer yet.

Pass one `--base-url` for each replica. Repeating the option enables multiple
servers; passing it once retains the original single-server behavior.

First validate alignment, workload, and maximum prompt size without contacting
the model server or writing output:

```bash
uv run --frozen python scripts/score_wmt23_gemba_v2.py \
  --model Qwen/Qwen3.5-35B-A3B-FP8 \
  --base-url http://127.0.0.1:8000/v1 \
  --base-url http://127.0.0.1:8001/v1 \
  --dry-run
```

Run a small two-judgment pilot in a disposable, separate output directory:

```bash
uv run --frozen python scripts/score_wmt23_gemba_v2.py \
  --model Qwen/Qwen3.5-35B-A3B-FP8 \
  --base-url http://127.0.0.1:8000/v1 \
  --base-url http://127.0.0.1:8001/v1 \
  --language-pair en-de \
  --system AIRC \
  --num-judgements 2 \
  --max-tokens 4096 \
  --limit 12 \
  --max-inflight-judgements 4 \
  --output-dir gemba-v2-pilot
```

For the full paper-style run, score all three language pairs and all MTME
systems, including the reference/human outputs required by a complete
source-based MTME score file:

```bash
nohup uv run --frozen python scripts/score_wmt23_gemba_v2.py \
  --model Qwen/Qwen3.5-35B-A3B-FP8 \
  --base-url http://127.0.0.1:8000/v1 \
  --base-url http://127.0.0.1:8001/v1 \
  --max-tokens 4096 \
  --max-inflight-judgements 128 \
  --output-dir gemba-v2-scores \
  > gemba-v2-full.log 2>&1 &
echo $! > gemba-v2-full.pid
```

`--max-inflight-judgements` is a hard per-endpoint generation budget. The
scorer sends every judgment as an independent `n=1` API request, so the same
value works for every `--num-judgements` setting. With the settings above, each
endpoint runs at most 128 requests, for at most 256 generating judgments across
both GPUs. A segment's successful judgments are flushed immediately under
`judgements/`; only a failed judgment is retried. Once all ten judgments for a
segment are present, the scorer aggregates them and appends its completed raw
segment record.
The scorer assigns complete documents to endpoints with a deterministic,
workload-balanced partition. Every system and segment belonging to one
document therefore reaches the same server and can reuse that server's prefix
cache. Requests are also queued in document groups to keep matching prefixes
close together before cache pressure can evict them. A single scorer process
owns all output files, retries, and exports.

Follow the full run with:

```bash
tail -f gemba-v2-full.log
```

The full run contains 68,130 segments and 681,300 independent `n=1` API
requests. Successful judgments are flushed immediately to
`gemba-v2-scores/judgements/*.jsonl`; completed segments are written to
`gemba-v2-scores/raw/*.jsonl`. Rerun the identical command to retain completed
judgments and retry only unfinished ones. A
manifest prevents accidentally mixing model, prompt, dataset, or sampling
configurations in one output directory. The replica URLs and count may change
between resumptions because they do not change the scores. Failures are logged
separately under `errors/`.

Before scoring, the default `--context-preflight documents` uses vLLM's
`/tokenize` endpoint and actual chat template to check the largest rendered
prompt for each document, reserving the default 4,096 `--max-tokens` within the
model limit. Use `--context-preflight all` for an exhaustive check of all
68,130 prompts.

When a language pair is complete, the scorer exports the aggregate and ten
individual-run metric files under:

```text
gemba-v2-scores/mtme/wmt23/metric-scores/LANGUAGE_PAIR/
```

Add the aggregate result to the reproduced table with:

```bash
uv run --frozen python scripts/reproduce_table3.py \
  --permutations 0 \
  --extra-metric-dir gemba-v2-scores/mtme \
  --output table3-with-gemba-v2.txt
```

If scoring is already complete but exports need to be recreated, rerun the
batch scorer with the same model/configuration and add `--export-only`.

## Test

```bash
uv run --frozen python scripts/reproduce_table3_test.py
uv run --frozen python scripts/score_wmt23_gemba_v2_test.py
```
