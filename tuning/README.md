# Coordinate-descent tuning for ALFM / CARP / DAML / MAN

Tunes the four ported review-aware models on the Amazon-2023 5-core datasets
(`musical`, `baby`, `cellphone`) by **coordinate descent**: start from constructor
defaults, sweep one parameter at a time, lock the value with the lowest **validation
RMSE**, move on. A JSONL cache (`cd_log.jsonl`) means no config is trained twice, so
runs are resumable — just re-run the same command.

## Setup

Use the `integrate` conda env (cornac editable + torch/gensim/sklearn). CARP/DAML/MAN
init embeddings from `../GoogleNews-vectors-negative300.bin.gz`; ALFM uses LDA topics
(no embedding file). Models auto-detect CUDA if a GPU torch build is installed.

## Usage

Run from the repo root so `import cornac` and `import tuning` resolve:

```bash
conda activate integrate
cd C:\Users\nguye\integrate\cornac

python -m tuning.run --model CARP --dataset musical      # one combo
python -m tuning.run --all --max-train 200000            # 4 models x 3 datasets
```

Key flags: `--epochs` (search fit, default 20), `--final-epochs` (final refit),
`--max-train N` (cap #train rows), `--w2v <path>`, `--seed`, `--quiet`, `--force`.

## Outputs (`tuning/results/<dataset>/<Model>/`)

- `cd_log.jsonl` — every evaluated config + val RMSE (the dedup cache).
- `best.json` — locked params, val/test RMSE-MAE, full CD trace, saved model path.
- `best_model/` — trained best model (`Recommender.save`). Reload with
  `cornac.models.CARP.load("<...>/best_model/CARP")`.
- `../summary.csv` — one row per (model, dataset).

## Editing the search

Grids and sweep order live in `search_spaces.py` (`SPACES[<Model>]["order"]`) plus fixed
context (`embedding_size=300`, `max_doc_length=500`, ...). Defaults are read live from each
model's constructor.

## Notes

- Splits: `cornac/data/amazon_2023/benchmark/5core/timestamp_w_his/*.csv`; reviews:
  `cornac/data/amazon_2023/<dataset>/review.txt`. `exclude_unknowns=True`.