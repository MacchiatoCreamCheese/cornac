"""Load the Amazon-2023 5-core splits + review text and build a Cornac eval method.

Data layout (under ``cornac/data/amazon_2023/``):
  - splits:  ``benchmark/5core/timestamp_w_his/<Name>.{train,valid,test}.csv``
             header ``user_id,parent_asin,rating,timestamp,history``
  - reviews: ``<short>/review.txt`` -> ``user_id \t parent_asin \t text``
             (lines may be wrapped in double quotes)
"""

import csv
import os
import random

# short name -> (benchmark file stem, review subfolder)
DATASETS = {
    "musical": "Musical_Instruments",
    "baby": "Baby_Products",
    "cellphone": "Cell_Phones_and_Accessories",
}

# This file lives at <repo>/tuning/; by default data is at <repo>/data/amazon_2023.
# Override for other machines/remotes with the CORNAC_TUNING_DATA env var or, from the
# CLI runners, --data-root (both should point at the amazon_2023 dir that contains
# benchmark/5core/... and <dataset>/review.txt).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_ROOT = os.path.join(_HERE, "..", "data", "amazon_2023")
DATA_ROOT = os.environ.get("CORNAC_TUNING_DATA", _DEFAULT_DATA_ROOT)


def set_data_root(path):
    """Point the loaders at a different amazon_2023 data dir (CLI --data-root)."""
    global DATA_ROOT
    if path:
        DATA_ROOT = os.path.abspath(os.path.expanduser(path))


def _split_path(name, split):
    stem = DATASETS[name]
    fname = {"train": "train", "val": "valid", "test": "test"}[split]
    return os.path.join(
        DATA_ROOT, "benchmark", "5core", "timestamp_w_his", f"{stem}.{fname}.csv"
    )


def _review_path(name):
    return os.path.join(DATA_ROOT, name, "review.txt")


def load_uir(name, split):
    """Return a list of (user_id, item_id, rating) tuples for a split."""
    data = []
    with open(_split_path(name, split), encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # user_id,parent_asin,rating,timestamp,history
        for row in reader:
            if len(row) < 3:
                continue
            data.append((row[0], row[1], float(row[2])))
    return data


def load_reviews(name):
    """Return a list of (user_id, item_id, review_text) triples."""
    triples = []
    with open(_review_path(name), encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line[0] == '"':
                line = line[1:]
            parts = line.split("\t", 2)
            if len(parts) < 3:
                continue
            user, item, text = parts[0], parts[1], parts[2]
            if text.endswith('"'):
                text = text[:-1]
            triples.append((user, item, text))
    return triples


def make_eval_method(name, max_train=None, seed=123, verbose=False):
    """Build a Cornac ``BaseMethod`` from the provided 5-core splits + reviews.

    Parameters
    ----------
    name: str
        One of ``musical``, ``baby``, ``cellphone``.
    max_train: int, optional
        If set, randomly subsample the training interactions to this many rows
        (the big-data knob). Validation/test are untouched; unknown users/items
        introduced by subsampling are dropped via ``exclude_unknowns=True``.
    seed: int
        Seed for subsampling and Cornac's internal RNG.
    """
    from cornac.data import ReviewModality
    from cornac.data.text import BaseTokenizer
    from cornac.eval_methods import BaseMethod

    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASETS)}.")

    train = load_uir(name, "train")
    val = load_uir(name, "val")
    test = load_uir(name, "test")

    if max_train is not None and max_train < len(train):
        rng = random.Random(seed)
        train = rng.sample(train, max_train)

    review_modality = ReviewModality(
        data=load_reviews(name),
        tokenizer=BaseTokenizer(stop_words="english"),
        max_vocab=50000,
        max_doc_freq=0.7,
    )

    eval_method = BaseMethod.from_splits(
        train_data=train,
        test_data=test,
        val_data=val,
        fmt="UIR",
        rating_threshold=4.0,
        exclude_unknowns=True,
        review_text=review_modality,
        seed=seed,
        verbose=verbose,
    )
    return eval_method
