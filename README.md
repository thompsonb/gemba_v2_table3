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

sudo apt-get install python3.12-dev
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
command computes all SPA, acc-t, and average values while skipping the slow
significance-cluster calculation. Per-task rank annotations are omitted in
this mode because ordinary ranks are not the paper's significance clusters.
The average column retains a rank recalculated over the 29 displayed metrics.
Omit `--permutations 0` to use the script's default of 1,000 significance-test
permutations when cluster ranks among the stored metrics are needed.

The script prints flushed progress messages to standard error before and after
loading each language pair and evaluating each of the six tasks. During an SPA
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

Reusable model launch commands are stored as `.txt` files under
`run_scripts/`. Run from that directory, they create the sibling `../runs/`
directory and write scorer outputs there. If you use the inline commands below
from the repository root instead, create that directory first:

```bash
mkdir -p runs
```

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
  --output-dir runs/gemba-v2-pilot
```

For the full paper-style run, score all three language pairs and all MTME
systems in one direct `n=10` run. This includes the reference/human outputs
required by a complete source-based MTME score file:

```bash
nohup uv run --frozen python scripts/score_wmt23_gemba_v2.py \
  --model Qwen/Qwen3.5-35B-A3B-FP8 \
  --base-url http://127.0.0.1:8000/v1 \
  --base-url http://127.0.0.1:8001/v1 \
  --num-judgements 10 \
  --temperature 0.4 \
  --max-tokens 4096 \
  --seed 0 \
  --max-inflight-judgements 128 \
  --output-dir runs/gemba-v2-all-qwen35-t04-n10-4k \
  > runs/gemba-v2-all-qwen35-t04-n10-4k.log 2>&1 &
echo $! > runs/gemba-v2-all-qwen35-t04-n10-4k.pid
```

`--max-inflight-judgements` is a hard per-endpoint generation budget. The
scorer sends every judgment as an independent `n=1` API request, so the same
value works for every `--num-judgements` setting. With the settings above, each
endpoint runs at most 128 requests, for at most 256 generating judgments across
both GPUs. A segment's successful judgments are flushed immediately under
`judgements/`; only a failed judgment is retried. Completed segment aggregates
are rebuilt in memory from those saved judgments and are not stored in a
second, duplicative format.
`--seed` names a deterministic sampling run. Each request seed is a stable hash
of the segment key, this base seed, and the judgment index. Reusing a base seed
reproduces or resumes the same judgments; changing it selects a fresh set rather
than shifting the previous run's seeds by one.
The scorer assigns complete documents to endpoints with a deterministic,
workload-balanced partition. Every system and segment belonging to one
document therefore reaches the same server and can reuse that server's prefix
cache. Requests are also queued in document groups to keep matching prefixes
close together before cache pressure can evict them. A single scorer process
owns all output files and retries.

Follow the full run with:

```bash
tail -f runs/gemba-v2-all-qwen35-t04-n10-4k.log
```

The full run contains 68,130 segments and 681,300 independent `n=1` API
requests. Successful judgments are flushed immediately to
`runs/gemba-v2-all-qwen35-t04-n10-4k/judgements/*.jsonl`. After a language pair is
complete, its JSONL file is atomically replaced by a verified `*.jsonl.zip`
archive. The uncompressed JSONL remains untouched until the temporary ZIP has
been closed and CRC-checked, so interruption during compression cannot destroy
the resumable input. The archives are the sole successful-result store after
completion and are read directly when resuming. Rerun the identical command to
rebuild completion state and retry only unfinished judgments. A
manifest prevents accidentally mixing model, prompt, dataset, or sampling
configurations in one output directory. The replica URLs and count may change
between resumptions because they do not change the scores. Failed judgments are
reported in the main log but are not persisted separately; their absent keys in
`judgements/` cause them to be retried on the next run.

Before scoring, the default `--context-preflight all` uses vLLM's `/tokenize`
endpoint and actual chat template to check all 68,130 rendered prompts while
reserving the default 4,096 `--max-tokens` within the model limit. The legacy
`--context-preflight documents` spelling is retained as an exhaustive alias;
UTF-8 or character length cannot safely identify the largest tokenized prompt.

Add the saved judgments directly to the reproduced table with:

```bash
uv run --frozen python scripts/reproduce_table3.py \
  --permutations 0 \
  --gemba-output-dir runs/gemba-v2-all-qwen35-t04-n10-4k \
  --allow-incomplete \
  --output table3-with-gemba-v2.txt
```

Repeat `--gemba-output-dir` to place several locally generated rows at the top
of one jointly ranked table:

```bash
uv run --frozen python scripts/reproduce_table3.py \
    --permutations 0 \
    --gemba-output-dir runs/gemba-v2-all-qwen35-122b-a10b-fp8-t04-n10-4k \
    --gemba-output-dir runs/gemba-v2-all-qwen38-27b-t04-n10-4k \
    --gemba-output-dir runs/gemba-v2-all-gemma4-31b-t04-n10-4k \
    --gemba-output-dir runs/gemba-v2-all-nemotron35-lightning-30b-a3b-bf16-t04-n10-4k \
    --gemba-output-dir runs/gemba-v2-all-qwen35-35b-a3b-fp8-t04-n10-4k/ \
    --allow-incomplete \
    --output table3-all-local-models.txt
```

The local rows retain their manifest-derived names, are ordered by decreasing
average at the top, and are followed by one separator. Their displayed ranks
are still global ranks against every row in the combined table.

The Table 3 script reads the compressed judgment files, applies GEMBA's
aggregation function in memory, and adds the resulting source-based segment
metric directly to MTME. If a historical or interrupted run has fewer than the
requested judgments for a segment, it aggregates those available; a segment
with no judgments is estimated as its system mean plus its segment mean minus
the language-pair mean, capped at zero. If an entire language pair has no saved
judgments, `--allow-incomplete` leaves its task values as `-` and computes the
local row's average as `-`. The script reports the exact backoff counts. The
local row name is derived from the model recorded in the run manifest; the
command above produces
`gemba-v2-qwen3.5-35b-a3b-fp8-rrwa(n=10)[noref]`. For comparison, the table also
includes the published `gemba-v2-gpt-4.1-mini-rrwa[noref]` Table 3 row. That
official row uses the paper's reported aggregate values because its underlying
segment judgments are not in the WMT23 MTME bundle. No derived metric-score
files are written.

`--allow-incomplete` is required for the checked-in historical archives because
some segments contain fewer than ten successful judgments. Omit the flag for a
new complete run so the table command fails instead of silently applying
backoffs to unfinished data.

## Test

```bash
uv run --frozen python scripts/reproduce_table3_test.py
uv run --frozen python scripts/score_wmt23_gemba_v2_test.py
```
