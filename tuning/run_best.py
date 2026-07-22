"""Run each faithful model ONCE with its best-guess config on one dataset.

The picks are each paper's own chosen values for the swept knobs; every pinned
value comes from ``search_spaces`` (same factory the tuner uses).

Examples
--------
    python -m tuning.run_best                          # full musical, 10 epochs each
    python -m tuning.run_best --max-train 1500 --epochs 2   # quick smoke
    python -m tuning.run_best --models CARP,MAN --epochs 30

Results are appended to ``tuning/results/best_run_<dataset>.jsonl`` -- one JSON
line per model, written IMMEDIATELY after that model finishes (crash-safe; rerun
skips already-logged models unless --force).
"""

import argparse
import json
import os
import time

import numpy as np

from . import datasets
from . import search_spaces

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")

# Best-guess values for the SWEPT knobs only (the papers' own chosen configs).
# Everything pinned lives in search_spaces.SPACES[<model>]["fixed"].
BEST_PICKS = {
    # Official released config (A=5, K=5, f=5) behind the paper's Musical result.
    "ALFM": {"num_factors": 5, "n_topics": 5},
    # CARP paper's final config: M=5, lambda=0.5, tau=3, k=25.
    "CARP": {"num_aspect": 5, "gama": 0.5, "itr_routing": 3, "latent_dim": 25},
    # Neu-Review-Rec reference values; batch 8 bounds the (bs,f,L,L) attention memory.
    "DAML": {
        "learning_rate": 2e-3, "dropout_rate": 0.5, "id_embedding_size": 32,
        "weight_decay": 1e-3, "batch_size": 8,
    },
    # MAN paper Sec. 5.4 config (batch 128 / gamma 50 / 4 heads already pinned).
    "MAN": {
        "learning_rate": 6e-3, "dropout_rate": 0.5, "id_embedding_size": 32,
        "n_filters": 100,
    },
}

# Cheap -> expensive, so results land early.
DEFAULT_ORDER = ["ALFM", "CARP", "DAML", "MAN"]


def _load_done(out_path):
    done = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["model"]] = rec
    return done


def _append(out_path, rec):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _prepare(model_name, base, eval_method, dataset, max_train, seed):
    """Model-specific precompute (mirrors tuning/run.py)."""
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
        print(f"  word2vec: {n_oov}/{vocab.size} OOV")

    if model_name == "ALFM":
        # Share the tuner's ATM disk cache (same key scheme as tuning/run.py).
        atm_dir = os.path.join(RESULTS_DIR, dataset, "ALFM", "atm_cache")
        os.makedirs(atm_dir, exist_ok=True)
        fname = "atm_A%d_K%d_it%d_mt%s_seed%s.npz" % (
            base.num_aspects, base.n_topics, base.tm_iterations, max_train, seed
        )
        fpath = os.path.join(atm_dir, fname)
        if os.path.exists(fpath):
            data = np.load(fpath)
            base.init_params[base._atm_key()] = {k: data[k] for k in data.files}
            print(f"  ATM: loaded from cache ({fname})")
        else:
            atm = base.build_atm(eval_method.train_set)
            np.savez_compressed(fpath, **atm)
            base.init_params[base._atm_key()] = atm
            print(f"  ATM: fitted and cached ({fname})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="musical", choices=list(datasets.DATASETS))
    ap.add_argument("--data-root", default=None,
                    help="amazon_2023 data dir (overrides CORNAC_TUNING_DATA / the "
                    "repo-relative default). Use on remotes where data lives elsewhere.")
    ap.add_argument("--models", default=",".join(DEFAULT_ORDER),
                    help="Comma-separated subset of ALFM,CARP,DAML,MAN.")
    ap.add_argument("--epochs", type=int, default=10,
                    help="Training epochs per model (paper-length runs need a GPU).")
    ap.add_argument("--max-train", type=int, default=None,
                    help="Cap #train interactions (quick passes). None = full data.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--w2v", default=None,
                    help="Pretrained word2vec path (default: search_spaces.DEFAULT_W2V).")
    ap.add_argument("--allow-random-embeddings", action="store_true",
                    help="Permit random word embeddings when the w2v file is missing "
                    "(won't reproduce paper numbers; for quick smokes only). Otherwise "
                    "a missing file errors.")
    ap.add_argument("--out", default=None,
                    help="Output JSONL (default: tuning/results/best_run_<dataset>.jsonl).")
    ap.add_argument("--force", action="store_true", help="Re-run already-logged models.")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--set", default=None, metavar="K=V[,K=V]",
        help="Override pick values for the models being run, e.g. --set batch_size=32. "
        "Ints/floats parsed automatically. Use with a single --models entry.",
    )
    args = ap.parse_args()

    datasets.set_data_root(args.data_root)

    overrides = {}
    if args.set:
        for kv in args.set.split(","):
            k, v = kv.split("=", 1)
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            overrides[k.strip()] = v

    from cornac.eval_methods import rating_eval
    from cornac.metrics import RMSE, MAE

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in BEST_PICKS]
    if unknown:
        ap.error(f"unknown model(s): {unknown}")

    out_path = args.out or os.path.join(
        RESULTS_DIR, f"best_run_{args.dataset}.jsonl"
    )
    done = _load_done(out_path)
    verbose = not args.quiet

    print(f"Loading dataset '{args.dataset}' (max_train={args.max_train})...")
    eval_method = datasets.make_eval_method(
        args.dataset, max_train=args.max_train, seed=args.seed, verbose=verbose
    )
    print(
        f"  users={eval_method.total_users} items={eval_method.total_items} "
        f"train={eval_method.train_set.num_ratings} "
        f"val={eval_method.val_set.num_ratings} test={eval_method.test_set.num_ratings}"
    )

    results = []
    for name in models:
        if name in done and not args.force:
            rec = done[name]
            print(f"\n=== {name}: already logged (rmse={rec['test_rmse']:.4f}) -- skip "
                  f"(--force to redo) ===")
            results.append(rec)
            continue

        picks = dict(BEST_PICKS[name], **overrides)
        print(f"\n=== {name}  epochs={args.epochs}  picks={picks} ===")
        base = search_spaces.build_base_model(
            name, max_iter=args.epochs, seed=args.seed, w2v_path=args.w2v,
            verbose=verbose, allow_random_embeddings=args.allow_random_embeddings,
        )
        _prepare(name, base, eval_method, args.dataset, args.max_train, args.seed)
        model = base.clone(picks)

        t0 = time.time()
        model.fit(eval_method.train_set, eval_method.val_set)
        scores = rating_eval(
            model, [RMSE(), MAE()], eval_method.test_set, verbose=verbose
        )[0]
        secs = time.time() - t0

        rec = {
            "model": name,
            "dataset": args.dataset,
            "picks": picks,
            "epochs": args.epochs,
            "max_train": args.max_train,
            "seed": args.seed,
            "test_rmse": scores[0],
            "test_mae": scores[1],
            "seconds": round(secs, 1),
        }
        _append(out_path, rec)  # logged immediately -- crash-safe
        results.append(rec)
        print(f"  {name}: test RMSE={scores[0]:.4f}  MAE={scores[1]:.4f}  ({secs:.0f}s)")

    print("\n===== Summary =====")
    print(f"{'model':<6} {'RMSE':>8} {'MAE':>8} {'seconds':>9}")
    for rec in results:
        print(f"{rec['model']:<6} {rec['test_rmse']:>8.4f} {rec['test_mae']:>8.4f} "
              f"{rec['seconds']:>9.1f}")
    print(f"\nResults logged to {out_path}")


if __name__ == "__main__":
    main()
