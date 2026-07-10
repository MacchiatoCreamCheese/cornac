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
"""Example for DAML (Dual Attention Mutual Learning) on Amazon Digital Music."""

import cornac
from cornac.datasets import amazon_digital_music
from cornac.eval_methods import RatioSplit
from cornac.data import ReviewModality
from cornac.data.text import BaseTokenizer

feedback = amazon_digital_music.load_feedback()
reviews = amazon_digital_music.load_review()

review_modality = ReviewModality(
    data=reviews,
    tokenizer=BaseTokenizer(stop_words="english"),
    max_vocab=4000,
    max_doc_freq=0.5,
)

ratio_split = RatioSplit(
    data=feedback,
    test_size=0.1,
    val_size=0.1,
    exclude_unknowns=True,
    review_text=review_modality,
    verbose=True,
    seed=123,
)

# For reproducibility, set pretrained_w2v_path to a word2vec/GloVe/fastText file.
model = cornac.models.DAML(
    embedding_size=300,
    id_embedding_size=32,
    n_filters=100,
    kernel_size=3,
    max_doc_length=500,
    batch_size=8,
    max_iter=10,
    learning_rate=0.002,
    pretrained_w2v_path=None,
    verbose=True,
    seed=123,
)

cornac.Experiment(
    eval_method=ratio_split,
    models=[model],
    metrics=[cornac.metrics.RMSE(), cornac.metrics.MAE()],
).run()
