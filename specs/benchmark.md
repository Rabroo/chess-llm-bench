# Spec: Chess LLM Benchmark

## Goal

Benchmark 19 large language models on chess move quality across ~5.8 million positions, measuring spatial reasoning ability as a proxy for genuine reasoning vs pattern matching. Primary research question: *"Can AI language models trained on text learn to reason spatially, as measured by chess performance?"*

## Inputs / Outputs

**Inputs:**
- `data/{easy,medium,hard,extreme}.json` — ~5.8M chess positions with Lc0 evaluations
- `config/config.yaml` — models, prompt formats, paths, thresholds
- Ollama instance with 19 models pulled and running

**Outputs:**
- `results/evaluations.jsonl` — one record per job: model, position, move chosen, centipawn loss, correctness
- `results/plots/*.png` — performance charts by model, difficulty, prompt format
- `results/metrics/*.csv` — aggregated metrics for analysis

## Steps / Logic

1. **Build dataset** (`scripts/build_dataset.py`) — load Lichess puzzle CSV, validate positions, split into difficulty tiers
2. **Lc0 evaluation** (`scripts/precompute_lc0_batch.py`) — GPU batch inference to get centipawn evaluations and best moves for all positions
3. **Pull models** (`scripts/pull_models.py`) — ensure all configured Ollama models are available
4. **Generate jobs** (`scripts/generate_jobs.py`) — create job queue in SQLite: every combination of position × model × prompt format (~330M jobs)
5. **Run workers** (`scripts/run_workers.py`) — parallel workers claim jobs, query Ollama, score moves, write results
6. **Generate plots** (`scripts/generate_plots.py`) — produce charts and CSVs from results

## Prompt Formats

- `fen_only` — just the FEN string
- `pgn+fen` — move history plus FEN
- `cot` — chain-of-thought, asks model to reason step by step

## Evaluation Metric

**Centipawn Loss (CPL)** — difference between the Lc0-evaluated best move and the model's chosen move, in centipawns. Lower is better. A threshold of 50 CPL is used to classify a move as "correct".

## Correction Loop

After a model makes a bad move (CPL > threshold), it is told it was wrong and asked again. A paired control group gets the same follow-up position without feedback. This tests self-correction as evidence of reasoning vs memorisation.

## Edge Cases

- Model returns illegal move — mark as failed, record error
- Model times out — retry up to `max_retries` times, then fail
- Duplicate jobs — detected via hash, skipped on insert
- OOM during job generation — handled by batch insertion (10k jobs at a time)
- SQLite file descriptor exhaustion — handled by single connection per batch

## Dependencies

| Dependency | Purpose |
|-----------|---------|
| Ollama | LLM inference server |
| Lc0 + ONNX model | Chess position evaluation |
| python-chess | FEN/PGN parsing, move validation |
| onnxruntime-gpu | GPU batch inference for Lc0 |
| SQLite | Job queue and deduplication |
| pytest | Test suite |

## Hardware

- GPU: RTX 5080 (for Lc0 ONNX batch inference)
- Storage: ~130GB for jobs DB, ~250GB for models split across main and shared NTFS partition

## Reproducibility

- Random seed: `42` (set in config)
- Lc0 model: `network.onnx` at path set in `config.yaml`
- All model versions pinned by Ollama tag
- Dataset source: Lichess puzzle database (`data/lichess_puzzles.csv`)
