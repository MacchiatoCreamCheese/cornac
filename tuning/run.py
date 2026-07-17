"""CLI for coordinate-descent hyperparameter tuning.

Examples
--------
    python -m tuning.run --model CARP --dataset musical
    python -m tuning.run --model DAML --dataset baby --max-train 200000 --epochs 20
    python -m tuning.run --all --max-train 200000

Outputs (under ``tuning/results/``):
    <dataset>/<Model>/cd_log.jsonl   every evaluated config (dedup cache; resumable)
    <dataset>/<Model>/best.json      locked params + val RMSE + test RMSE/MAE + trace
    summary.csv                      one row per (model, dataset)
"""

import argparse
import csv
import json
import os

import numpy as np

from . import datasets
from . import search_spaces
from .coordinate_descent import run_cd, final_eval

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")

MODELS = ["ALFM", "CARP", "DAML", "MAN"]


def tune_one(
    model_name, dataset, epochs, final_epochs, max_train, seed, w2v, verbose,
    force=False,
):
    print(f"\n===== {model_name} on {dataset} =====")

    out_dir = os.path.join(RESULTS_DIR, dataset, model_name)
    best_json = os.path.join(out_dir, "best.json")
    # Combo-level resume: a completed run wrote best.json -> skip it (use --force to redo).
    # (Within a run, mid-sweep resume is automatic via the cd_log.jsonl cache: completed
    # fits are cache hits, and an interrupted candidate simply re-trains from scratch.)
    if os.path.exists(best_json) and not force:
        with open(best_json, encoding="utf-8") as f:
            best = json.load(f)
        print(f"  [skip] already done (best.json exists): "
              f"test_rmse={best.get('test_rmse')}. Use --force to redo.")
        return best

    eval_method = datasets.make_eval_method(
        dataset, max_train=max_train, seed=seed, verbose=verbose
    )
    print(
        f"  users={eval_method.total_users} items={eval_method.total_items} "
        f"train={eval_method.train_set.num_ratings} "
        f"val={eval_method.val_set.num_ratings} test={eval_method.test_set.num_ratings}"
    )

    base = search_spaces.build_base_model(
        model_name, max_iter=epochs, seed=seed, w2v_path=w2v, verbose=verbose
    )

    # Precompute the pretrained embedding matrix ONCE (vocab is fixed per dataset)
    # and inject it via init_params, so each candidate fit reuses it instead of
    # reloading the 3M-vector word2vec file from disk.
    if getattr(base, "pretrained_w2v_path", None):
        from cornac.models.carp.w2v_utils import build_w2v_matrix

        vocab = eval_method.train_set.review_text.vocab
        mat, n_oov = build_w2v_matrix(
            vocab,
            path=base.pretrained_w2v_path,
            emb_type=base.pretrained_w2v_type,
            word_dim=base.embedding_size,
            seed=seed,
        )
        base.init_params["pretrained_word_embeddings"] = mat
        print(f"  word2vec: {n_oov}/{vocab.size} OOV (loaded once, cached in memory)")

    # ALFM: the ATM topic model depends only on the docs + (num_aspects, n_topics),
    # not on num_factors. Precompute it ONCE per distinct n_topics value in the sweep
    # (num_aspects is pinned) and inject each under its atm key, so the coordinate
    # descent reuses them across all factor candidates instead of refitting Gibbs.
    # Each fitted ATM (~15 min of Gibbs on a full dataset) is persisted to disk
    # immediately, so an interrupted sweep never recomputes finished topic models.
    if model_name == "ALFM":
        alfm_grid = dict(search_spaces.SPACES["ALFM"]["order"])
        k_values = sorted(set(alfm_grid.get("n_topics", [])) | {base.n_topics})
        atm_dir = os.path.join(out_dir, "atm_cache")
        os.makedirs(atm_dir, exist_ok=True)
        print(f"  precomputing ATM for K in {k_values} (disk-cached in {atm_dir})...")
        orig_k = base.n_topics
        for K in k_values:
            base.n_topics = K
            # Cache key: everything the ATM depends on besides the docs themselves
            # (dataset dir + max_train + seed pin the docs; A/K/tm_iterations the fit).
            fname = "atm_A%d_K%d_it%d_mt%s_seed%s.npz" % (
                base.num_aspects, K, base.tm_iterations, max_train, seed
            )
            fpath = os.path.join(atm_dir, fname)
            if os.path.exists(fpath):
                data = np.load(fpath)
                atm = {k: data[k] for k in data.files}
                print(f"    K={K}: loaded from cache ({fname})")
            else:
                atm = base.build_atm(eval_method.train_set)
                np.savez_compressed(fpath, **atm)  # persist immediately
                print(f"    K={K}: fitted and cached ({fname})")
            base.init_params[base._atm_key()] = atm
        base.n_topics = orig_k

    order = search_spaces.SPACES[model_name]["order"]
    defaults = search_spaces.defaults(model_name)
    if verbose:
        print(f"  epochs/fit={epochs}  search space:")
        for p, vals in order:
            print(f"    {p}: {vals}")

    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "cd_log.jsonl")

    best_params, trace = run_cd(
        base, order, defaults, eval_method, cache_path, verbose=verbose
    )
    metrics = final_eval(
        base,
        best_params,
        eval_method,
        final_epochs=final_epochs,
        save_dir=os.path.join(out_dir, "best_model"),
        verbose=verbose,
    )

    best = {
        "model": model_name,
        "dataset": dataset,
        "best_params": best_params,
        "search_val_rmse": trace[-1]["val_rmse"],
        **metrics,
        "trace": trace,
    }
    with open(os.path.join(out_dir, "best.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    _append_summary(model_name, dataset, best_params, trace[-1]["val_rmse"], metrics)
    print(
        f"  DONE {model_name}/{dataset}: "
        f"val_rmse={trace[-1]['val_rmse']:.4f} "
        f"test_rmse={metrics['test_rmse']:.4f} test_mae={metrics['test_mae']:.4f}"
    )
    return best


def _append_summary(model_name, dataset, best_params, val_rmse, metrics):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "summary.csv")
    header = [
        "model", "dataset", "val_rmse", "test_rmse", "test_mae", "best_params",
    ]
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(
            [
                model_name,
                dataset,
                f"{val_rmse:.4f}",
                f"{metrics['test_rmse']:.4f}",
                f"{metrics['test_mae']:.4f}",
                json.dumps(best_params),
            ]
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=MODELS, help="Model to tune.")
    ap.add_argument("--dataset", choices=list(datasets.DATASETS), help="Dataset.")
    ap.add_argument("--all", action="store_true", help="Run all models x all datasets.")
    ap.add_argument("--epochs", type=int, default=10, help="Epochs per search fit (iRev default=10).")
    ap.add_argument(
        "--final-epochs",
        type=int,
        default=None,
        help="Epochs for the final refit (defaults to --epochs).",
    )
    ap.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Cap #train interactions (big-data subsample). None = full 5-core.",
    )
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--w2v",
        default=None,
        help="Path to pretrained word2vec (default: search_spaces.DEFAULT_W2V).",
    )
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-run combos even if a completed best.json already exists.",
    )
    args = ap.parse_args()

    verbose = not args.quiet
    if args.all:
        combos = [(m, d) for d in datasets.DATASETS for m in MODELS]
    elif args.model and args.dataset:
        combos = [(args.model, args.dataset)]
    elif args.dataset:  # all models on one dataset
        combos = [(m, args.dataset) for m in MODELS]
    elif args.model:  # one model on all datasets
        combos = [(args.model, d) for d in datasets.DATASETS]
    else:
        ap.error("provide --dataset and/or --model, or use --all")

    for model_name, dataset in combos:
        tune_one(
            model_name,
            dataset,
            epochs=args.epochs,
            final_epochs=args.final_epochs,
            max_train=args.max_train,
            seed=args.seed,
            w2v=args.w2v,
            verbose=verbose,
            force=args.force,
        )


if __name__ == "__main__":
    main()
