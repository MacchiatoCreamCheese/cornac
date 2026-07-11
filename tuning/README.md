# Coordinate-descent tuning for ALFM / CARP / DAML / MAN

Finds the best hyperparameters for the four ported review-aware models on the
Amazon-2023 5-core datasets (`musical`, `baby`, `cellphone`) by **coordinate descent**:
start from the constructor defaults, sweep one parameter with the others locked, lock
its best value (lowest **validation RMSE**), then move to the next parameter.

A persistent JSONL cache (`cd_log.jsonl`) keyed on the full tuned-parameter set means a
configuration is **never trained twice** — the default, and any value already locked,
is a cache hit rather than a retrain. Runs are therefore resumable: re-running continues
where it left off.

## Setup

Use the `integrate` conda env (cornac built editable, torch/gensim/sklearn installed).
The word-based models (CARP/DAML/MAN) initialize embeddings from
`../GoogleNews-vectors-negative300.bin.gz` by default; ALFM uses LDA topics and needs no
embedding file. For GPU, install a CUDA build of torch in the env (the models auto-detect
CUDA); `musical` runs fine on CPU.

## Usage

This package lives at `<repo>/cornac/tuning/`. Run it from the cornac repo root
(`C:\Users\nguye\integrate\cornac`) so `import cornac` and `import tuning` both resolve
from the working directory — no `PYTHONPATH` needed:

```bash
conda activate integrate
cd C:\Users\nguye\integrate\cornac

# one model / one dataset
python -m tuning.run --model CARP --dataset musical

# subsample big datasets for the search (the big-data knob)
python -m tuning.run --model DAML --dataset cellphone --max-train 200000

# everything (4 models x 3 datasets)
python -m tuning.run --all --max-train 200000
```

Key flags: `--epochs` (per search fit, default 20), `--final-epochs` (final refit),
`--max-train N` (cap #train interactions), `--w2v <path>`, `--seed`, `--quiet`,
`--force` (re-run a combo whose `best.json` already exists).

## Resuming after a disconnect

Just re-run the same command. Two levels of resume:
- **Across combos** (`--all`): any (model, dataset) whose `best.json` exists is skipped.
- **Within a run**: the per-combo `cd_log.jsonl` cache is replayed — every completed fit is
  a cache hit, so coordinate descent fast-forwards to exactly the parameter/value it was on
  when interrupted, then re-trains that one candidate from scratch (no mid-training resume).
  The default and any already-locked value are never retrained.

## Outputs (`cornac/tuning/results/`)

- `<dataset>/<Model>/cd_log.jsonl` — every evaluated config + val RMSE (the dedup cache).
- `<dataset>/<Model>/best.json` — locked params, search val RMSE, **test** RMSE/MAE, full
  coordinate-descent trace, and the saved model path.
- `<dataset>/<Model>/best_model/` — the trained best model, saved via Cornac's
  `Recommender.save` (`.pkl` + `.pt` state dict). Reload with e.g.
  `cornac.models.CARP.load("<...>/best_model/CARP")`.
- `summary.csv` — one row per (model, dataset).

## Editing the search

All grids and the sweep order live in `search_spaces.py` (`SPACES[<Model>]["order"]`), plus
the fixed context (`embedding_size=300`, `max_doc_length=500`, etc.). Defaults are read live
from each model's constructor, so the baseline always matches the real defaults.

## Notes

- Splits come from `cornac/data/amazon_2023/benchmark/5core/timestamp_w_his/*.csv`; reviews
  from `cornac/data/amazon_2023/<dataset>/review.txt`. `exclude_unknowns=True`, so val/test
  interactions with users/items unseen in (subsampled) train are dropped.
- `cellphone` (2.5 GB) and `baby` (0.8 GB) are large; prefer `--max-train` for the search,
  then a final full-data refit of the locked config (raise/omit `--max-train` for the final
  run, or re-run that config manually).
