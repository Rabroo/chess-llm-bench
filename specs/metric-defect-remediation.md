# Spec: Metric Defect Remediation (post-audit review)

**Created:** 2026-08-13
**Author:** Ryan Brew
**Status:** Open — no fixes applied yet
**Supersedes nothing.** This is a follow-up to `specs/artefact-fixes.md` (the
2026-05-07 audit). That audit fixed A1–A5 in `by_model.csv`, `summary.json`
and `docs/FINDINGS.md` — but did **not** propagate the corrections to the
model-family aggregation or to hypothesis tests H3 and H4, and introduced no
chance-corrected baseline for direction accuracy.

---

## Goal

Close six defects that would undermine the benchmark if published as-is.

None of these require re-running LLM inference, Stockfish, or Lc0. All are
postprocessing over the existing `results/evaluations.jsonl` (526,662 records),
exactly like the May audit.

**What is not in question:** every headline number in `docs/FINDINGS.md`
reconciles against `results/metrics/by_model.csv` to within rounding. This was
re-verified on 2026-08-13 across all eight headline figures (legality 97.847
vs 97.85 claimed; clamped CPL 701.6 vs 702; direction t0/t50/t100/t200 =
44.18/46.19/61.75/79.06 vs 44.5/46.1/61.6/78.9; wp_loss 310.7 vs 311; T3 v2
0.519 vs 0.520). The retry experiment reconciles exactly (10,665 records,
7,212 legal = 67.6%). The defects below are about **metric construction and
stale propagation**, not arithmetic errors.

---

## Inputs / Outputs

**Inputs**
- `results/evaluations.jsonl` — 526,662 records (local only; not in git)
- `results/metrics/*.csv`, `results/metrics/summary.json` — current outputs
- `src/metrics.py`, `src/utils.py` — computation under repair

**Outputs**
- Corrected `results/metrics/by_model_family.csv` (new columns, correct groups)
- Corrected `summary.json` H3/H4 blocks
- New `results/metrics/direction_baselines.csv` — per-threshold class priors
- New `results/metrics/cpl_buckets.csv` — blunder-rate breakdown with CIs
- Updated `docs/FINDINGS.md` Findings 1, 3, 5 and Summary Table
- New tests in `tests/test_metric_defects.py`

---

## Defect register

Severity: **P0** blocks publication · **P1** materially misleads · **P2** cosmetic

---

### D1 — Direction accuracy is compared against an invalid baseline (P0)

**Status:** Root cause verified in code. Magnitude needs raw data to confirm.

`docs/FINDINGS.md` Finding 3 reports direction accuracy against a stated
**"Random baseline (3-class) = 33%"** at every threshold. That baseline is only
valid when the three classes are balanced. They are not.

`_compute_direction_correct` (`src/metrics.py:258-271`) applies the *same*
threshold to both the model eval and the Stockfish eval:

```python
me_dir = np.where(model_eval > threshold, "W",
                   np.where(model_eval < -threshold, "B", "E"))
sf_dir = np.where(stockfish_eval > threshold, "W",
                   np.where(stockfish_eval < -threshold, "B", "E"))
correct = (me_dir == sf_dir)
```

As `threshold → ∞`, both sides collapse to `"E"` for every record and accuracy
converges to **100% by construction**, independent of model skill. So the
reported climb 44.5% → 46.1% → 61.6% → 78.9% is partly a mechanical artefact of
raising the threshold, not evidence of improving model judgment.

This matters concretely because the ground truth is heavily imbalanced. Our own
methodology section states **92% of Stockfish ground-truth evals lie within
±300 cp**. If the ±200 "equal" class holds ~80% of positions, then a degenerate
model answering `"E"` unconditionally scores ~80% — **above** the 78.9%
all-model average currently presented as a positive result.

**Why it matters:** Finding 3's strongest claim ("at a 2-pawn threshold, models
agree with Stockfish 79% of the time") could invert to "models fail to beat the
majority-class baseline." This is the first thing a reviewer will test.

**Fix**
1. Compute the ground-truth class prior at each threshold in
   `DIRECTION_THRESHOLDS_CP` and emit `direction_baselines.csv` with columns
   `threshold, pct_W, pct_B, pct_E, majority_class, majority_baseline_acc`.
2. Report **Cohen's κ** and **balanced accuracy** per (model, threshold)
   alongside raw accuracy. κ corrects for chance agreement given the actual
   marginals; raw accuracy does not.
3. Replace the "Random baseline 33%" column in Finding 3 with the
   majority-class baseline per threshold.
4. If κ ≤ 0 at ±200, say so explicitly and rewrite Finding 3's conclusion.

**Verification**
- `test_direction_baseline_matches_priors` — synthetic df, known class split,
  assert `majority_baseline_acc` equals the majority prior.
- `test_degenerate_predictor_scores_baseline` — a model emitting constant 0 cp
  must score ≈ the majority baseline and **κ ≈ 0** at every threshold.
- `test_kappa_zero_for_random_predictor` — shuffled predictions give κ ≈ 0.

---

### D2 — Direction accuracy is non-monotonic and reverses model ranking (P0)

**Status:** Verified from `by_model.csv`.

Same root cause as D1, different symptom. Per-model accuracy across thresholds:

| model | t0 | t50 | t100 | t200 |
|---|---|---|---|---|
| gemma3:4b | **10.4** | 52.2 | 75.8 | 91.0 |
| llama3.3:70b | 58.2 | 34.3 | **21.9** | 42.2 |
| gemma4:31b | 52.2 | 38.8 | **31.8** | 61.7 |
| qwen2.5:14b | 58.2 | 47.6 | 66.5 | 80.0 |
| deepseek-r1:14b | 41.8 | 53.0 | 76.7 | 91.8 |

gemma3:4b swings 7× (10.4% → 75.8%) on identical underlying predictions.
llama3.3:70b and gemma4:31b both dip below the 33% figure and then *recover* at
±200 — non-monotonic, which a well-posed accuracy metric should not be. The
metric is measuring the interaction between each model's **output scale** and
the threshold, not its ability to identify the stronger side.

Consequence: Finding 3's headline claims — "llama3.3:70b is systematically
wrong", "the largest model in the study is anti-correlated with truth", and the
whole "play vs reason decoupling" narrative — are **threshold-contingent**. At
t0, llama3.3:70b (58.2%) *outranks* deepseek-r1:14b (41.8%), the exact reverse
of the reported ranking.

**Competing explanation that must be ruled out first:** a sign or
perspective bug. If a model reports evaluations from the side-to-move's
perspective rather than White's, its sign inverts on every Black-to-move
position (~50% of records) and produces exactly this signature. The two models
that dip below random (llama3.3:70b, gemma4:31b) are the prime suspects.
"70B models are anti-correlated with chess truth" is an extraordinary claim; a
parser bug is the ordinary one, and it must be eliminated before publication.

**Fix**
1. Split direction accuracy by side-to-move (`_white_to_move_from_fen` already
   exists at `src/metrics.py:61`). If llama3.3:70b's accuracy is high on
   white-to-move and inverted on black-to-move, it is a perspective bug — fix
   the parser in `src/evaluator.py` and recompute. This is the decisive test.
2. Sample 50 raw llama3.3:70b and gemma4:31b eval responses and hand-check the
   parsed sign against the model's own words.
3. Report a scale-free rank measure (Spearman ρ between model eval and
   Stockfish eval per model) that does not depend on any threshold. Lead with
   this; keep the threshold table as a secondary robustness view.

**Verification**
- `test_direction_split_by_side_to_move` — synthetic perspective-inverted
  model scores high on white-to-move, near-zero on black-to-move.
- `test_spearman_rho_threshold_free` — ρ is unchanged when all evals are
  scaled by a positive constant (thresholded accuracy is not).

---

### D3 — Model family/size parsing merges distinct models (P0, live bug)

**Status:** Verified exactly. Arithmetic below is exact, not approximate.

`parse_model_info` (`src/utils.py:140-186`) has two independent faults.

**Fault A — incomplete family map.** Only five prefixes are recognised:

```python
family_map = {"qwen": "qwen", "llama": "llama", "mistral": "mistral",
              "phi": "phi", "gemma": "gemma"}
```

Missing: `deepseek`, `mixtral`, `solar`, `codellama`, `yi`, `command-r`,
`wizardlm`. **8 of 22 models fall into family `"unknown"`.**

**Fault B — size parser fails on non-integer tags.**

```python
size_clean = size_str.lower().replace("b", "")   # "e2b" -> "e2", "8x7b" -> "8x7"
try: size_b = int(size_clean)
except ValueError:
    try: size_b = float(size_clean)
    except ValueError: pass                       # size_b stays 0
```

`e2b`, `e4b`, and `8x7b` all silently parse to **0.0 billion parameters**.

`aggregate_by_model_family` (`src/metrics.py:454`) then groups by
`["model_family", "model_size_b"]`, so any two models colliding on that pair
are **averaged into one row**. Since no single model exceeds 24,000 records,
any row above that is a merge. Three rows are:

| row in `by_model_family.csv` | n | actually contains |
|---|---|---|
| `gemma`, size `0.0` | 47,962 | gemma4:e2b (23,963) + gemma4:e4b (23,999) |
| `unknown`, size `7.0` | 46,935 | deepseek-r1:7b (22,970) + wizardlm2:7b (23,965) |
| `unknown`, size `34.0` | 47,982 | codellama:34b (24,000) + yi:34b (23,982) |

All three sums are exact. `mixtral:8x7b` also lands at `unknown/0.0`.

**Why it matters:** CodeLlama and Yi averaged together is not a family, and
**H4 ("larger models perform better within family") is computed over these
mislabelled groups** — see `src/metrics.py:677-682`. Any scaling claim derived
from this table is unsound.

**Fix**
1. Extend `family_map` to cover all 22 tags. Note ordering: check `codellama`
   before `llama` if prefix matching is kept, or switch to explicit exact-tag
   lookup (preferred — it fails loudly on an unknown tag).
2. Parse sizes with a regex that handles `e2b`/`e4b` (Gemma "effective"
   variants) and `8x7b` (MoE). Record MoE models with both total and active
   parameter counts; they are not comparable to dense models on a single axis
   and should be excluded from scaling regressions.
3. **Raise on `family == "unknown"` or `size_b == 0`** rather than silently
   bucketing. A silent 0.0 is what caused this.
4. Add an assertion in `aggregate_by_model_family` that each output row's
   `job_id` count is ≤ the max per-model count, and that group count equals the
   distinct model count.

**Verification**
- `test_parse_model_info_all_configured_tags` — every tag in `config.yaml`
  yields a non-`unknown` family and non-zero size.
- `test_parse_model_info_effective_and_moe_tags` — `gemma4:e2b`→2,
  `gemma4:e4b`→4, `mixtral:8x7b`→(56 total / 12.9 active) or explicit MoE flag.
- `test_codellama_not_matched_as_llama` — regression guard on prefix ordering.
- `test_family_rows_are_one_model_each` — no output row exceeds max per-model
  record count.
- `test_unknown_family_raises` — unrecognised tag raises, not defaults.

---

### D4 — H3, H4 and the family table still use pre-audit metrics (P1)

**Status:** Verified in code and in `summary.json`.

The May audit added corrected columns (`t2_cpl_clamped`, `t2_wp_loss`,
`t3_score_v2`, `t1_abs_error_excl_mate`, `t2_legal_attempted`,
`t1_direction_correct_t{0,50,100,200}`) and wired them into H1 and H2 via the
`primary_metric` / `metrics` pattern. **H3, H4 and
`aggregate_by_model_family` were missed.**

| site | code | uses | should use |
|---|---|---|---|
| H3 | `metrics.py:661` | `t3_score` | `t3_score_v2` |
| H4 | `metrics.py:682` | `t1_absolute_error`, `t2_cpl`, `t3_score` | `t1_abs_error_excl_mate`, `t2_cpl_clamped`, `t3_score_v2` |
| family agg | `metrics.py:454` | `t1_absolute_error`, `t2_cpl`, `t2_legal`, `t3_score` | corrected equivalents |

Confirmation from the shipped artefacts: H3's values in `summary.json`
(`easy 0.4176, medium 0.4342, hard 0.4029, extreme 0.3809`) match the **v1**
`t3_score` column in `by_difficulty.csv` — the matcher documented in A3 as
missing 70% of theme classes. And `by_model_family.csv`'s `t2_legal` column
sits at ~0.66, i.e. it is still the **A4 buggy legality denominator** that the
audit replaced with 97.85%.

So the repo simultaneously publishes 97.85% legality in `by_model.csv` and
64.6%-flavoured legality in `by_model_family.csv`. That internal contradiction
is trivially discoverable.

**Fix:** apply the H1/H2 `primary_metric`/`metrics` pattern to H3 and H4;
add corrected columns to the family aggregation; keep the legacy columns
alongside for transparency, suffixed `_legacy`.

**Verification**
- `test_h3_uses_v2_theme_score` — H3 reports `primary_metric == "t3_score_v2"`.
- `test_h4_uses_corrected_metrics` — H4 reports clamped CPL, not `t2_cpl`.
- `test_family_table_has_corrected_columns` — asserts presence of all five
  corrected columns.
- `test_no_buggy_legality_in_published_csvs` — no shipped CSV carries a
  legality column whose all-model mean is < 0.90.

---

### D5 — Clamped CPL is saturated; model ranking rests on ~2% of scale (P1)

**Status:** Verified from `by_model.csv`.

Per-model **median** clamped CPL runs **901–980** against a clamp of 1000.
More than half of all legal moves sit within 5–10% of the ceiling. Cross-model
standard deviation is **22.5 on a 0–1000 scale**.

The headline "gemma4:31b best at 634, gemma4:e2b worst at 727" is a 93-point
gap on a metric that is pinned near its cap, with 39.1% of moves exactly at it.
Mechanically, `mean_clamped_cpl ≈ 1000 × P(blunder) + (small residual)` — the
metric is a **blunder-rate proxy in centipawn clothing**, and presenting it as
a continuous quality score overstates its resolution.

**Fix**
1. Report **blunder rate** (`P(CPL ≥ 1000)`) as the primary T2 quality metric —
   it is what the number actually encodes, and it is directly interpretable.
2. Attach bootstrap 95% CIs to every per-model T2 figure. With ~24k records per
   model the CIs will be tight, but the point is to show the reader that a
   634-vs-727 gap on a saturated metric is a weak ordering.
3. Emit `cpl_buckets.csv` with the five buckets already in Finding 2, per model,
   with counts — so readers can see the bimodality directly rather than
   inferring it from a mean.
4. Keep median alongside mean everywhere; a mean of 702 against a median of
   ~950 is itself informative and currently unreported per-model.

**Verification**
- `test_blunder_rate_matches_bucket_counts` — blunder rate equals the
  `CPL ≥ 1000` bucket fraction.
- `test_cpl_ci_overlap_flagged` — models with overlapping 95% CIs are marked
  `rank_tied = True`.

---

### D6 — Legality headline measures the harness, not the model (P1)

**Status:** Known (A6), under-weighted in the writeup.

Finding 1 leads with **97.85% legality**. Per A6 this is *post-rescue*: on
`cot`/`fen_only`/`pgn+fen`, `src/worker.py:208-222` silently (a) scans the
response for any other legal SAN/UCI token, then (b) re-prompts with the full
legal-move list. And `move_only` supplies the legal-move list up front. The
single-shot rate is unrecoverable from stored data.

Publishing "models produce a legal move 98% of the time" as the lead finding
invites exactly one question — *what was the first-attempt rate?* — which
currently has no answer. The supporting evidence points the other way: on the
10,665 records where the rescue still failed, an explicit re-prompt with the
answer set fixed only **67.6%**.

**Fix**
1. Relabel throughout as **"harness-assisted legality (≤2 attempts)"**. Never
   present the bare 98% without that qualifier.
2. Instrument `src/worker.py` to record `t2_move_attempt_1`,
   `t2_legal_attempt_1`, and `t2_rescue_path` for future runs.
3. Re-run a **single-shot legality probe** on a stratified subsample
   (suggest 200 positions × 22 models × 3 combined formats ≈ 13,200 calls,
   a few GPU-hours) with rescue disabled. That yields a defensible
   single-shot number without re-running the full 526k benchmark.
4. Demote legality from Finding 1. Lead with the bimodal CPL result — it is
   the genuine contribution and the finding least affected by D1–D6.

**Verification**
- `test_worker_records_first_attempt` — rescue path populates attempt-1 fields.
- `test_rescue_disabled_flag` — `rescue_enabled=False` returns the raw first
  parse unmodified.

---

### D7 — T2 ground-truth engine is ambiguous and weak (P2, document)

`docs/FINDINGS.md` states T2 best-move/CPL truth is **Lc0 @ 800 nodes**, but
`results/evaluations_retried.jsonl` names its field `stockfish_best_move`, and
`retried_cpl` is null in all 10,665 records. The artefacts disagree with the
documentation about which engine produced T2 truth.

800 nodes is also a weak reference — and we already report that the two engines
disagree on the best move on **~67%** of the overlap.

**Fix:** rename the field to its actual source; state node/depth settings and
engine versions in `summary.json` provenance (see D8); soften "ground truth" to
"reference engine" for T2; add a sensitivity check re-scoring a 1,000-position
subsample at higher Lc0 node counts to show the ranking is stable.

---

### D8 — Doc/data drift and missing provenance (P2)

| item | doc | data |
|---|---|---|
| raw uncapped mean CPL | 4,786 (`FINDINGS.md`) | 4,735.0 (`summary.json`) |
| all-model legality | 97.85% | 97.88% |
| direction t50 | — | `summary.json` carries **two** values: `t1_direction_accuracy_t50 = 0.46262` and `by_threshold["t50"] = 0.46135` |

Also: `summary.json` has no `generated_at`, git SHA, config snapshot, or engine
versions. The only committed run logs are March **pilot** runs (50 and 5
positions per tier, 2 models, one at depth 15 not 22) — there is no run log for
the actual 526k run, so its date is only recoverable from commit archaeology.

**Fix:** single source of truth for each number (docs generated from
`summary.json`, never hand-typed); reconcile the duplicate t50; stamp
`generated_at`, git SHA, engine versions and config snapshot into
`summary.json` in `scripts/generate_plots.py`.

---

## Steps / Logic

Ordered. D1–D3 gate publication; do them first.

1. **D3 first** — it is a live code bug and cheapest to fix. Correct
   `parse_model_info`, add the guard assertions, regenerate
   `by_model_family.csv`. ~1 hour.
2. **D2 decisive test** — split direction accuracy by side-to-move. If a
   perspective bug exists, this reveals it immediately and changes what D1
   needs to say. Do not proceed to D1 write-up before this resolves. ~1 hour.
3. **D1** — compute class priors and κ, emit `direction_baselines.csv`, then
   rewrite Finding 3 against whatever the numbers actually show. ~2 hours.
4. **D4** — propagate corrected metrics into H3/H4 and the family table. ~1 hour.
5. **D5** — blunder rate, bootstrap CIs, `cpl_buckets.csv`. ~2 hours.
6. **D6** — relabel legality; optionally run the single-shot probe. ~1 hour +
   GPU time.
7. **D7/D8** — provenance stamping, field rename, doc regeneration. ~1 hour.
8. Rewrite `docs/FINDINGS.md` Findings 1/3/5 and the Summary Table. Restructure
   to lead with the bimodal CPL result.
9. Full `pytest` run; commit; push.

---

## Edge cases

- **Positions where Stockfish eval is mate-encoded** (7% of records) — must be
  excluded from Spearman ρ, or the ±16,000 values dominate the rank
  correlation. Reuse the `MATE_SCORE_THRESHOLD_CP = 9000` constant.
- **Models with missing records** — deepseek-r1:7b is short ~1,030 of 24,000
  (~4.3%). Bootstrap CIs must resample within model, not across.
- **κ undefined** when a model emits a single class for every position;
  return `NaN` explicitly and surface it rather than coercing to 0.
- **MoE parameter counts** — `mixtral:8x7b` has 56B total / ~12.9B active.
  Excluded from scaling regressions; flagged, not silently assigned one number.
- **gemma4:e2b / e4b** are "effective parameter" variants; their real memory
  footprint differs from dense models of the same nominal size. Document in the
  scaling section rather than plotting them on the same axis.
- **Ties in `rank_tied`** — CIs overlapping pairwise is not transitive; report
  as a set of statistically indistinguishable models, not a chain.
- **Rerunning the family aggregation** must not silently drop a model — assert
  output group count equals distinct model count (22).
- **`qwen2.5:72b`** is commented out in `config.yaml` (OOM at 32GB RAM/16GB
  VRAM) but listed as pulled in `docs/CHANGES.md`. Reconcile the docs; it is
  not in the 22.

---

## Dependencies

- Python 3.11+, `pandas`, `numpy` (already in `requirements.txt`)
- **`scipy`** — new. Needed for `scipy.stats.spearmanr` and bootstrap CIs.
  Add to `requirements.txt`.
- `sklearn.metrics.cohen_kappa_score` and `balanced_accuracy_score`, **or** a
  hand-rolled κ (~15 lines) to avoid a large new dependency. Prefer
  hand-rolled; it is trivial and keeps the dependency surface small.
- `results/evaluations.jsonl` — **local to the Linux PC only, not in git.**
  All of this work must happen there.
- D6 step 3 (single-shot probe) additionally needs Ollama + all 22 models
  resident, and ~13,200 inference calls.

---

## Acceptance criteria

1. `pytest` green, including all new tests in `tests/test_metric_defects.py`.
2. No shipped CSV contains a pre-audit metric column without a `_legacy` suffix.
3. `by_model_family.csv` has exactly 22 rows, one per model, none `unknown`,
   none size 0.
4. `summary.json` carries `generated_at`, git SHA, engine versions, config
   snapshot.
5. `direction_baselines.csv` exists; Finding 3 quotes the majority-class
   baseline and κ, not the 33% figure.
6. Every number in `docs/FINDINGS.md` is traceable to a CSV or `summary.json`
   key — none hand-typed.
7. The D2 side-to-move split is resolved and its outcome documented, whichever
   way it lands.

---

## Note on scope

D1–D8 are metric-construction and propagation defects. The data collection
itself — 526,662 records at 99.75% of planned coverage, no duplicate
`(model, position_id, prompt_format)` keys, seeded and stratified sampling —
is sound, and nothing here requires re-collecting it.

The **bimodal move-quality result** (17.8% near-perfect, 39.1% catastrophic,
flat across 2B→70B) is the study's real contribution and is the finding least
disturbed by these fixes. The restructured writeup should lead with it.
