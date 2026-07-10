# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Build a pretrained word-embedding matrix aligned to a Cornac vocabulary.

This is a faithful port of the iRev benchmark's ``w2v.npy`` construction
(``iRev/preprocess_data/pro_data.py``): each vocabulary token is looked up in a
pretrained word2vec/GloVe/fastText model; out-of-vocabulary tokens receive a
random vector drawn from ``uniform(-1, 1)``. The only difference is that we
iterate over Cornac's ``review_text.vocab`` instead of iRev's own vocabulary.
"""

import warnings

import numpy as np


def build_w2v_matrix(vocab, path=None, emb_type="word2vec", word_dim=300, seed=None):
    """Return a ``(vocab.size, word_dim)`` embedding matrix.

    Parameters
    ----------
    vocab: :obj:`cornac.data.text.Vocabulary`
        The review vocabulary (its ``idx2tok`` defines row order; special tokens
        ``<PAD>``/``<UNK>``/``<BOS>``/``<EOS>`` occupy the first rows).

    path: str, optional, default: None
        Path to a pretrained embedding file. If ``None``, all rows are random
        (``uniform(-1, 1)``) and a warning is emitted -- results will not match
        the paper, but the model still runs offline.

    emb_type: str, {'word2vec', 'glove', 'fasttext'}, default: 'word2vec'
        Format of the pretrained file. 'word2vec' expects the binary
        ``GoogleNews-vectors-negative300.bin``; 'fasttext' a ``.vec`` text file;
        'glove' a GloVe file already converted to word2vec text format
        (``gensim.scripts.glove2word2vec``).

    word_dim: int, default: 300
        Embedding dimension. Must match the pretrained file when ``path`` is set.

    seed: int, optional
        Seed for the random OOV vectors (and the full-random fallback).
    """
    rng = np.random.RandomState(seed)

    if path is None:
        warnings.warn(
            "No pretrained_w2v_path provided; initializing word embeddings "
            "randomly. Results will not reproduce the paper numbers.",
            stacklevel=2,
        )
        matrix = rng.uniform(-1.0, 1.0, (vocab.size, word_dim)).astype("float32")
        return matrix, vocab.size

    import gensim

    binary = emb_type == "word2vec"
    kwargs = {"binary": binary}
    if emb_type == "fasttext":
        kwargs["limit"] = 999999
    kv = gensim.models.KeyedVectors.load_word2vec_format(path, **kwargs)

    if kv.vector_size != word_dim:
        raise ValueError(
            "Pretrained embedding dim (%d) != word_dim (%d)"
            % (kv.vector_size, word_dim)
        )

    matrix = np.empty((vocab.size, word_dim), dtype="float32")
    n_oov = 0
    for idx, tok in enumerate(vocab.idx2tok):
        if tok in kv:
            matrix[idx] = kv[tok]
        else:
            n_oov += 1
            matrix[idx] = rng.uniform(-1.0, 1.0, (word_dim,))
    return matrix, n_oov


def build_doc_matrices(review_text, num_users, num_items, max_doc_length):
    """Build per-user and per-item document token-id matrices.

    Reproduces iRev's ``userDoc2Index``/``itemDoc2Index``: for every user (item),
    concatenate the token sequences of all of its reviews into a single document,
    then truncate/zero-pad to ``max_doc_length``.

    Parameters
    ----------
    review_text: :obj:`cornac.data.ReviewModality`
        A built review modality (``group_by=None``) exposing ``sequences``,
        ``user_review`` and ``item_review``.

    num_users, num_items: int
        Number of users/items in the training set.

    max_doc_length: int
        Fixed document length (tokens); iRev uses 500.

    Returns
    -------
    user_doc, item_doc: numpy.ndarray of shape (num_users, L) / (num_items, L)
        Integer token-id matrices, zero-padded at the end.
    """
    sequences = review_text.sequences

    def _docs(review_group, n):
        docs = np.zeros((n, max_doc_length), dtype="int64")
        for idx in range(n):
            review_ids = review_group.get(idx, {})
            doc = []
            for review_idx in review_ids.values():
                doc.extend(sequences[review_idx])
                if len(doc) >= max_doc_length:
                    break
            doc = doc[:max_doc_length]
            docs[idx, : len(doc)] = doc
        return docs

    user_doc = _docs(review_text.user_review, num_users)
    item_doc = _docs(review_text.item_review, num_items)
    return user_doc, item_doc
