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
        # Faithful CARP (official https://github.com/WHUIR/CARP). Pinned to the
        # paper/official defaults; sweeps the four knobs the paper itself ablates.
        # Note: CARP no longer has id_embedding_size/weight_decay/dropout_rate --
        # it uses RMSProp (no decay), DropConnect keep-prob, and scalar user/item
        # bias instead of id embeddings. max_doc_length is 300 (not the shared 500).
        "fixed": dict(
            embedding_size=300,
            pretrained_w2v_type="word2vec",
            max_doc_length=300,
            n_filters=50,
            kernel_size=3,
            itr_self_attn=2,
            lambda_1=0.8,
            rating_threshold=3.0,
            dropout_keep_prob=0.9,
            batch_size=100,
            learning_rate=1e-3,
        ),
        "order": [  # coordinate descent, constructor default listed first for cache de-dup
            ("num_aspect", [5, 3, 7, 9]),                # M -- CARP Table 3
            ("gama", [0.5, 0.1, 0.3, 0.7, 1.0]),         # lambda -- CARP Fig 3(b)
            ("itr_routing", [3, 1, 2, 4]),               # tau -- CARP Table 4
            ("latent_dim", [25, 50, 100]),               # k -- paper "optimal in [25,100]"
        ],
    },
    "DAML": {
        # Faithful DAML (NFM head + additive review/id fusion). Grid grounded in the
        # DAML paper's own tuning ranges (p.6-7) and the Neu-Review-Rec reference config.
        # kernel_size=3 (paper "sliding window 3") and n_filters=100 (paper "100 kernels";
        # Neu-Review-Rec filters_num=100) are pinned. Each order lists the constructor
        # default first so the tuner cache de-dups the baseline.
        "fixed": dict(_TEXT_FIXED, kernel_size=3, n_filters=100),
        "order": [
            # paper tunes lr in {1e-5,2e-5,1e-3,2e-3}; Neu-Review-Rec uses 2e-3. The 1e-5/2e-5
            # end needs far more epochs than the benchmark budget, so keep the practical subset.
            ("learning_rate", [2e-3, 1e-3]),
            # paper searches dropout in {0.1,0.2,0.3,0.4} (DAML uses 0.2); Neu-Review-Rec 0.5.
            ("dropout_rate", [0.5, 0.2, 0.1, 0.3, 0.4]),
            # paper tunes latent dim in {8,16,32,64,128} (DAML uses 8); Neu-Review-Rec/iRev 32.
            ("id_embedding_size", [32, 8, 16, 64, 128]),
            # paper reg {0.001,0.01,...}; Neu-Review-Rec weight_decay 1e-3 (higher values blow up Adam WD).
            ("weight_decay", [1e-3, 1e-2]),
            # paper batch {50,100,150,200}; Neu-Review-Rec 128; smaller kept for the O(doc_len^2) memory.
            ("batch_size", [8, 32, 64, 128]),
        ],
    },
    "MAN": {
        # Faithful MAN (main+auxiliary nets, 3-step distillation; implemented from the
        # paper -- no official repo). Grid grounded in the paper's Sec. 5.4: MAN's own
        # values are pinned (batch 128, gamma/fc_dim 50, 4 heads, FF 128, kernel 3,
        # lambda/weight_decay 1e-3, RT length 50); the sweep covers the knobs the paper
        # itself searched. Word embeddings: shared 300-dim pipeline vectors for now
        # (paper states 64-dim pretrained -- revisit later). Constructor default first.
        "fixed": dict(
            _TEXT_FIXED,
            max_doc_length=1000,  # [MAN 5.4] max input text 1,000 (overrides _TEXT_FIXED's 500)
            kernel_size=3,
            fc_dim=50,
            att_hidden=64,
            n_heads=4,
            ff_dim=128,
            max_rt_length=50,
            batch_size=128,
            weight_decay=1e-3,
        ),
        "order": [
            ("learning_rate", [6e-3, 1e-3]),               # paper searched {1e-5,6e-5,1e-3,6e-3}; MAN uses 6e-3
            ("dropout_rate", [0.5, 0.1, 0.2, 0.3, 0.4]),   # paper searched {0.1..0.5}; MAN uses 0.5
            ("id_embedding_size", [32, 8, 16, 64]),        # paper searched {8,16,32,64}; MAN uses 32
            ("n_filters", [100, 50, 150, 200]),            # paper searched {50,100,150,200}; MAN uses 100
        ],
    },
    "ALFM": {
        # Faithful ALFM (official Java, WWW'18): two-stage ATM topic model + aspect-aware
        # latent-factor SGD. No word embeddings. Sweeps the paper-ablated knobs f and K;
        # Dirichlet priors, regs, aspect count, and topic-model iterations are pinned to the
        # official values. num_factors is swept before n_topics so all f candidates reuse a
        # single precomputed topic model (only a K change forces an ATM refit).
        "fixed": dict(
            num_aspects=5,
            alpha=0.1, beta=0.01, gamma=0.5,
            tm_iterations=120, tm_begin_save=100, tm_save_step=10,
            learn_rate=0.01, reg=0.5, weight_reg=0.01,
        ),
        "order": [
            ("num_factors", [5, 10, 15, 20, 25]),  # f -- ALFM Fig 2/3
            ("n_topics",    [5, 10, 15, 20, 25]),  # K -- ALFM Fig 2/3
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
