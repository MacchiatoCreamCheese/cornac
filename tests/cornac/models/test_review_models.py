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
"""Tests for the review-based models ported from iRev: ALFM, CARP, DAML, MAN.

The heavy PyTorch backend is optional, so the tests are skipped when torch is
not installed (mirroring the pattern used by the DMRL tests).
"""

import os
import random
import tempfile
import unittest

import numpy as np

try:
    import torch  # noqa: F401

    from cornac.data import ReviewModality
    from cornac.data.text import BaseTokenizer
    from cornac.eval_methods import RatioSplit
    from cornac.models import ALFM, CARP, DAML, MAN

    RUN = True
except ImportError:
    RUN = False


_WORDS = (
    "great good bad awful nice terrible love hate cheap expensive fast slow "
    "quality product sound music album track price value recommend "
    "battery screen easy hard fun boring classic modern".split()
)


def _synthetic_split(seed=42):
    rng = random.Random(seed)
    n_users, n_items = 25, 18
    feedback, reviews = [], []
    for u in range(n_users):
        # each user interacts with several items (dense enough to survive split)
        for it in rng.sample(range(n_items), rng.randint(6, 10)):
            rating = rng.randint(1, 5)
            text = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(8, 20)))
            feedback.append((f"u{u}", f"i{it}", float(rating)))
            reviews.append((f"u{u}", f"i{it}", text))

    review_modality = ReviewModality(
        data=reviews,
        tokenizer=BaseTokenizer(stop_words="english"),
        max_vocab=200,
        max_doc_freq=1.0,
    )
    return RatioSplit(
        data=feedback,
        test_size=0.1,
        val_size=0.1,
        exclude_unknowns=True,
        review_text=review_modality,
        verbose=False,
        seed=seed,
    )


class TestReviewModels(unittest.TestCase):
    def setUp(self):
        if not RUN:
            self.skipTest("PyTorch is not available")
        self.split = _synthetic_split()

    def _check_fitted(self, model):
        model.fit(self.split.train_set, self.split.val_set)

        # Score a single (user, item) pair.
        s = model.score(0, 0)
        self.assertTrue(np.isfinite(s))

        # Score all items for a user.
        all_scores = model.score(0)
        self.assertEqual(all_scores.shape[0], self.split.train_set.num_items)
        self.assertTrue(np.all(np.isfinite(all_scores)))

    def _check_save_load(self, model, cls):
        with tempfile.TemporaryDirectory() as tmp:
            path = model.save(tmp)
            self.assertTrue(os.path.exists(path.replace(".pkl", ".pt")))
            loaded = cls.load(path)
            before = model.score(0, 0)
            after = loaded.score(0, 0)
            self.assertAlmostEqual(before, after, places=4)

    def test_carp(self):
        model = CARP(
            embedding_size=16,
            id_embedding_size=8,
            n_filters=6,
            max_doc_length=40,
            batch_size=16,
            max_iter=2,
            verbose=False,
            seed=42,
        )
        self._check_fitted(model)
        self._check_save_load(model, CARP)

    def test_daml(self):
        model = DAML(
            embedding_size=16,
            id_embedding_size=8,
            n_filters=6,
            max_doc_length=40,
            batch_size=8,
            max_iter=2,
            verbose=False,
            seed=42,
        )
        self._check_fitted(model)
        self._check_save_load(model, DAML)

    def test_man(self):
        model = MAN(
            embedding_size=16,
            id_embedding_size=8,
            n_filters=6,
            fc_dim=8,
            max_doc_length=40,
            batch_size=16,
            max_iter=2,
            verbose=False,
            seed=42,
        )
        self._check_fitted(model)
        self._check_save_load(model, MAN)

    def test_alfm(self):
        model = ALFM(
            id_embedding_size=8,
            n_topics=8,
            max_doc_length=40,
            batch_size=16,
            max_iter=2,
            lda_max_iter=5,
            verbose=False,
            seed=42,
        )
        self._check_fitted(model)
        self._check_save_load(model, ALFM)


if __name__ == "__main__":
    unittest.main()
