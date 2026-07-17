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
"""RO / ORT / RT review-stream builder for the faithful MAN model.

MAN separates three review streams (paper Sec. 3, Table 1):

* **RT**  = ``t_ui`` -- the target user's review of the target item.
* **RO**  = ``t_u - t_ui`` -- the target user's reviews of *other* items.
* **ORT** = ``t_i - t_ui`` -- *other* users' reviews of the target item.

The main network consumes RO and ORT; the auxiliary network consumes RT. Each
stream is a single concatenated document (paper Eq. 1). RO/ORT are built per
(user, item) pair with the target review excluded. For a *test* pair (no training
review) nothing is excluded (RO = all the user's reviews, ORT = all the item's
reviews) and RT is unused -- exactly the paper's test-time setup (main net only).
"""

import numpy as np


def train_corpus_w2v(review_text, word_dim=64, seed=None, epochs=5):
    """Pretrain word2vec on the training reviews (paper Sec. 5.4: 64-dim pretrained).

    The paper cites a pretrained embedding method without naming a public 64-dim
    file, so we pretrain gensim word2vec on the training-review corpus itself and
    align the vectors to Cornac's vocabulary. Rows for tokens unseen by word2vec
    (special tokens, min-count casualties) are random uniform(-0.1, 0.1); row 0
    (<PAD>) is zeroed by the model.

    Returns a ``(vocab.size, word_dim)`` float32 matrix.
    """
    import gensim

    vocab = review_text.vocab
    idx2tok = vocab.idx2tok
    sentences = [
        [idx2tok[t] for t in seq if 0 <= t < vocab.size]
        for seq in review_text.sequences
    ]
    model = gensim.models.Word2Vec(
        sentences=sentences, vector_size=word_dim, window=5, min_count=1,
        workers=4, seed=seed if seed is not None else 1, epochs=epochs,
    )
    rng = np.random.RandomState(seed)
    matrix = rng.uniform(-0.1, 0.1, (vocab.size, word_dim)).astype("float32")
    for idx, tok in enumerate(idx2tok):
        if tok in model.wv:
            matrix[idx] = model.wv[tok]
    return matrix


class ReviewStreams:
    """Cache per-user / per-item review token sequences and assemble RO/ORT/RT batches."""

    def __init__(self, review_text, max_doc_length, max_rt_length):
        seqs = review_text.sequences
        self.max_doc_length = max_doc_length
        self.max_rt_length = max_rt_length

        # user_idx -> list[(item_idx, token_array)];  item_idx -> list[(user_idx, token_array)]
        self.user_items = {}
        for u, items in review_text.user_review.items():
            self.user_items[u] = [
                (i, np.asarray(seqs[r], dtype=np.int64)) for i, r in items.items()
            ]
        self.item_users = {}
        for i, users in review_text.item_review.items():
            self.item_users[i] = [
                (u, np.asarray(seqs[r], dtype=np.int64)) for u, r in users.items()
            ]
        # user_idx -> {item_idx: token_array}  (target-review lookup for RT)
        self.rt_lookup = {}
        for u, items in review_text.user_review.items():
            self.rt_lookup[u] = {
                i: np.asarray(seqs[r], dtype=np.int64) for i, r in items.items()
            }

    @staticmethod
    def _concat_excluding(entries, exclude_key, max_len):
        """Concatenate the token arrays of ``entries`` skipping ``exclude_key``, truncate."""
        parts = [tok for k, tok in entries if k != exclude_key]
        if not parts:
            return np.zeros(0, dtype=np.int64)
        doc = np.concatenate(parts)
        return doc[:max_len]

    def build_batch(self, u_arr, i_arr):
        """Return (RO, ORT, RT) int64 arrays for the batch of (user, item) pairs.

        RO/ORT: shape (bs, max_doc_length). RT: shape (bs, max_rt_length). Padded
        with 0 (the <PAD> id). The target review is excluded from RO and ORT.
        """
        bs = len(u_arr)
        ro = np.zeros((bs, self.max_doc_length), dtype=np.int64)
        ort = np.zeros((bs, self.max_doc_length), dtype=np.int64)
        rt = np.zeros((bs, self.max_rt_length), dtype=np.int64)

        for b in range(bs):
            u = int(u_arr[b])
            i = int(i_arr[b])

            doc_ro = self._concat_excluding(self.user_items.get(u, []), i, self.max_doc_length)
            ro[b, : len(doc_ro)] = doc_ro

            doc_ort = self._concat_excluding(self.item_users.get(i, []), u, self.max_doc_length)
            ort[b, : len(doc_ort)] = doc_ort

            trt = self.rt_lookup.get(u, {}).get(i)
            if trt is not None:
                trt = trt[: self.max_rt_length]
                rt[b, : len(trt)] = trt

        return ro, ort, rt
