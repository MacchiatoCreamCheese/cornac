"""Coordinate-descent search engine with a persistent dedup cache.

Sweeps one hyperparameter at a time (others locked), picks the value with the best
validation RMSE, locks it, and moves on. A JSONL cache keyed on the full tuned-param
set guarantees that a configuration is never evaluated twice -- in particular the
default (and any already-locked value) reappearing inside a later sweep is a cache
hit, not a retrain.
"""

import json
import os
import time


def _key(params):
    """Canonical, order-independent key for a tuned-parameter dict."""
    return json.dumps({k: params[k] for k in sorted(params)}, sort_keys=True)


def _load_cache(cache_path):
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cache[_key(rec["params"])] = rec["val_rmse"]
    return cache


def _append_cache(cache_path, params, val_rmse, seconds):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"params": params, "val_rmse": val_rmse, "seconds": seconds}
            )
            + "\n"
        )


def run_cd(base_model, order, defaults, eval_method, cache_path, verbose=True):
    """Run coordinate descent. Returns (best_params, trace).

    Parameters
    ----------
    base_model: cornac Recommender
        Base model carrying the FIXED context (embedding size, w2v, max_iter, ...).
        Tuned params are overridden per candidate via ``base_model.clone(params)``.
    order: list[(str, list)]
        Ordered (param_name, candidate_values) pairs.
    defaults: dict
        Starting (locked) values for the tuned params.
    eval_method: cornac BaseMethod
        Provides ``train_set``, ``val_set``, ``test_set``.
    cache_path: str
        JSONL file; results are appended and reused across runs.
    """
    from cornac.eval_methods import rating_eval
    from cornac.metrics import RMSE

    cache = _load_cache(cache_path)
    rmse = RMSE()

    # Upper bound on the number of NEW fits (the locked/default value in each sweep
    # is a cache hit): baseline + sum(len(values) - 1).
    n_fits_max = 1 + sum(max(len(v) - 1, 0) for _, v in order)
    counters = {"done": 0, "fit": 0, "t_fit": 0.0}
    t_start = time.time()

    def evaluate(params):
        k = _key(params)
        counters["done"] += 1
        if k in cache:
            if verbose:
                print(f"      cache-hit  RMSE={cache[k]:.4f}   {params}")
            return cache[k], True
        counters["fit"] += 1
        t0 = time.time()
        model = base_model.clone(params).fit(
            eval_method.train_set, eval_method.val_set
        )
        score = rating_eval(model, [rmse], eval_method.val_set)[0][0]
        secs = time.time() - t0
        counters["t_fit"] += secs
        cache[k] = score
        _append_cache(cache_path, params, score, secs)
        if verbose:
            avg = counters["t_fit"] / counters["fit"]
            print(
                f"      fit {counters['fit']}/{n_fits_max}  {secs:6.1f}s  "
                f"RMSE={score:.4f}  (avg {avg:.1f}s/fit)   {params}"
            )
        del model
        return score, False

    trace = []
    current = dict(defaults)

    # Baseline (all defaults) evaluated once.
    if verbose:
        print(f"  Baseline {current}")
    base_score, _ = evaluate(current)
    trace.append({"step": "baseline", "params": dict(current), "val_rmse": base_score})
    if verbose:
        print(f"  baseline val RMSE = {base_score:.4f}")

    for step, (param, values) in enumerate(order, 1):
        if verbose:
            locked = {k: current[k] for k in current if k != param}
            print(
                f"\n  ── step {step}/{len(order)}: sweep '{param}' over {values} "
                f"(locked: {locked}) ──"
            )
        best_val, best_score = current[param], None
        for v in values:
            cand = dict(current, **{param: v})
            score, _ = evaluate(cand)
            better = best_score is None or score < best_score
            if better:
                best_score, best_val = score, v
            if verbose:
                mark = " *best" if better else ""
                print(f"      {param}={v}: RMSE={score:.4f}{mark}")
        current[param] = best_val
        trace.append(
            {
                "step": param,
                "locked_value": best_val,
                "val_rmse": best_score,
                "params": dict(current),
            }
        )
        if verbose:
            print(f"  → locked {param}={best_val}  (val RMSE={best_score:.4f})")

    if verbose:
        print(
            f"\n  coordinate descent done: {counters['fit']} fits, "
            f"{time.time() - t_start:.0f}s total"
        )
    return current, trace


def final_eval(
    base_model,
    best_params,
    eval_method,
    final_epochs=None,
    save_dir=None,
    verbose=True,
):
    """Refit the locked config, evaluate on the TEST split, and (optionally) save
    the trained best model with Cornac's ``Recommender.save``.

    Returns a dict with test metrics and, if saved, ``model_path``.
    """
    from cornac.eval_methods import rating_eval
    from cornac.metrics import RMSE, MAE

    params = dict(best_params)
    model = base_model.clone(params)
    if final_epochs is not None:
        model.max_iter = final_epochs
    if verbose:
        print(f"  Final refit on train, params={params}, epochs={model.max_iter}")
    model.fit(eval_method.train_set, eval_method.val_set)
    scores = rating_eval(model, [RMSE(), MAE()], eval_method.test_set)[0]
    result = {"test_rmse": scores[0], "test_mae": scores[1]}

    if save_dir is not None:
        # The saved model carries its own doc caches + vocab_size, so it is
        # self-contained for scoring without the (large) train_set.
        model_path = model.save(save_dir, save_trainset=False)
        result["model_path"] = model_path
        if verbose:
            print(f"  Saved best model -> {model_path}")

    return result
