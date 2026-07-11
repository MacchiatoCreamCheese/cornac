"""Coordinate-descent search spaces and model factories.

Each model has:
  - ``FIXED``: hyperparameters held constant across the whole search.
  - ``ORDER``: an ordered list of ``(param_name, [candidate values])``. Coordinate
    descent sweeps them top-to-bottom; the first candidate that equals the current
    locked value (the default at step 1) is deduplicated by the cache, so the
    default is never retrained.

Defaults come from the model constructors and define the baseline config.
Edit the grids below to change the search.
"""

# Word2vec file used to initialize embeddings for the word-based models
# (CARP/DAML/MAN). ALFM uses LDA topics and ignores it. Overridable via CLI.
import os

# This file lives at <repo>/cornac/tuning/; the w2v file is at <repo>/GoogleNews-...
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_W2V = os.path.join(_HERE, "..", "..", "GoogleNews-vectors-negative300.bin.gz")


# Context shared by all word-based models during the search.
_TEXT_FIXED = {
    "embedding_size": 300,
    "max_doc_length": 500,
    "pretrained_w2v_type": "word2vec",
}


# Only values with a documented source are kept (survey Table 9, the model papers,
# or iRev defaults). Values I had merely interpolated were removed -- see
# SEARCH_SPACE_PROVENANCE.md. Parameters left with a single sourced value are pinned
# in "fixed" (they equal the constructor default) rather than swept.
SPACES = {
    "CARP": {
        # Pinned to CARP-paper window c=3 and iRev id/filters/batch.
        "fixed": dict(
            _TEXT_FIXED,
            kernel_size=3,
            id_embedding_size=32,
            n_filters=100,
            batch_size=128,
        ),
        "order": [
            ("learning_rate", [1e-3, 2e-3]),   # 1e-3 = CARP paper; 2e-3 = survey/iRev
            ("dropout_rate", [0.1, 0.5]),      # 0.1 = CARP paper (keep-prob 0.9); 0.5 = survey/iRev
            ("weight_decay", [1e-4, 1e-3]),    # survey T9
        ],
    },
    "DAML": {
        # kernel fixed 3 (paper), n_filters=100 (survey, no alternative) -> pinned.
        "fixed": dict(_TEXT_FIXED, kernel_size=3, n_filters=100),
        "order": [
            ("learning_rate", [1e-3, 2e-3]),   # survey T9 (paper's 1e-5/2e-5 excluded; see MD)
            ("dropout_rate", [0.2, 0.5]),      # 0.2 DAML paper, 0.5 survey/iRev
            ("id_embedding_size", [8, 32]),    # 8 DAML paper, 32 paper/survey
            ("weight_decay", [1e-4, 1e-3]),    # survey T9
            ("batch_size", [8, 32, 64, 128]),  # 8 iRev; 32/64/128 survey (large may OOM)
        ],
    },
    "MAN": {
        # MAN paper gave no numeric hyperparameters; dropout/id/filters/fc_dim have
        # only a single sourced value (survey/iRev) -> pinned.
        # kernel_size=3 (sliding window) and batch_size=128 pinned per user request.
        # NOTE: iRev's train.sh ran MAN with batch_size=32, not 128.
        "fixed": dict(
            _TEXT_FIXED,
            dropout_rate=0.5,
            id_embedding_size=32,
            n_filters=100,
            fc_dim=32,
            kernel_size=3,
            batch_size=128,
        ),
        "order": [
            ("learning_rate", [1e-3, 2e-3]),   # survey T9
            ("weight_decay", [1e-4, 1e-3]),    # survey T9
        ],
    },
    "ALFM": {
        # No word embeddings. dropout/n_topics/id have only a single sourced value
        # (survey/iRev) -> pinned. (ALFM paper's f=K=5 maps to latent dim, not our
        # LDA n_topics; iRev used LDA-32.)
        "fixed": {"lda_max_iter": 10, "dropout_rate": 0.5, "n_topics": 32, "id_embedding_size": 32},
        "order": [
            ("learning_rate", [1e-3, 2e-3]),   # survey T9
            ("weight_decay", [1e-4, 1e-3]),    # survey T9
            ("batch_size", [32, 64, 128]),     # survey T9
        ],
    },
}

# Which models consume a pretrained word2vec file.
_WORD_BASED = {"CARP", "DAML", "MAN"}


def build_base_model(model_name, max_iter, seed=123, w2v_path=None, verbose=False):
    """Instantiate a base model with fixed context wired in (defaults elsewhere)."""
    import cornac.models as M

    cls = getattr(M, model_name)
    fixed = dict(SPACES[model_name]["fixed"])
    kwargs = dict(fixed, max_iter=max_iter, seed=seed, verbose=verbose)

    if model_name in _WORD_BASED:
        path = w2v_path if w2v_path is not None else DEFAULT_W2V
        kwargs["pretrained_w2v_path"] = path if os.path.exists(path) else None

    return cls(**kwargs)


def defaults(model_name):
    """The starting (locked) values = first-column defaults from the constructor.

    We read them off a freshly constructed model so the baseline always matches
    the actual constructor defaults.
    """
    import cornac.models as M

    cls = getattr(M, model_name)
    probe = cls()
    return {p: getattr(probe, p) for p, _ in SPACES[model_name]["order"]}
