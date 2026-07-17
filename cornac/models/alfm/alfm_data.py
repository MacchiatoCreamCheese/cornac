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
"""Document/sentence builder for the faithful ALFM aspect-aware topic model.

The official ATM (``ALFM/src/topicmodel/Documents.java``) treats each training
review as a *document* made of *sentences*, and each sentence as a bag of word
ids. Sentences are lowercased, tokens shorter than 2 characters are dropped, and
the topic model builds its **own** vocabulary (separate from Cornac's review
vocab). The Java input is pre-segmented with ``||``; here we segment Cornac's raw
review text on sentence punctuation (``. ! ?``) or an explicit ``||`` if present.

This produces a flattened (CSR-style) document structure the Cython ATM sampler
consumes without any Python-object access in its inner loop.
"""

import re

import numpy as np

_SENT_SPLIT = re.compile(r"[.!?]+|\|\|")
_WORD = re.compile(r"[a-z0-9]+")


def _sentences(text):
    """Split raw review text into sentences, each a list of >=2-char word ids-to-be."""
    for chunk in _SENT_SPLIT.split(text):
        words = [w for w in _WORD.findall(chunk.lower()) if len(w) >= 2]
        if words:
            yield words


def build_documents(review_text, num_users, num_items):
    """Build the flattened ATM document structure from a Cornac review modality.

    Parameters
    ----------
    review_text: :obj:`cornac.data.ReviewModality`
        Built with ``group_by=None`` so that ``reviews`` (idx -> raw text) and
        ``user_review`` (user_idx -> {item_idx: review_idx}) are available.
    num_users, num_items: int
        Training user/item counts.

    Returns
    -------
    docs: dict
        Flattened arrays for the Cython sampler:
          - ``doc_user`` (M,), ``doc_item`` (M,)         int32
          - ``doc_sent_ptr`` (M+1,)                      int32  (CSR into sentences)
          - ``sent_word_ptr`` (S+1,)                     int32  (CSR into words)
          - ``words`` (nwords,)                          int32  (topic-model word ids)
          - ``vocab_size`` int, ``num_users``/``num_items`` int
    """
    word2id = {}
    doc_user, doc_item = [], []
    doc_sent_ptr = [0]
    sent_word_ptr = [0]
    words = []

    reviews = review_text.reviews  # idx -> raw text
    for user_idx, item_dict in review_text.user_review.items():
        for item_idx, review_idx in item_dict.items():
            text = reviews.get(review_idx, "")
            n_sent_before = len(sent_word_ptr) - 1
            for sent_words in _sentences(text):
                for w in sent_words:
                    wid = word2id.get(w)
                    if wid is None:
                        wid = len(word2id)
                        word2id[w] = wid
                    words.append(wid)
                sent_word_ptr.append(len(words))
            # Only register the document if it produced at least one sentence.
            if len(sent_word_ptr) - 1 > n_sent_before:
                doc_user.append(user_idx)
                doc_item.append(item_idx)
                doc_sent_ptr.append(len(sent_word_ptr) - 1)

    return {
        "doc_user": np.asarray(doc_user, dtype=np.int32),
        "doc_item": np.asarray(doc_item, dtype=np.int32),
        "doc_sent_ptr": np.asarray(doc_sent_ptr, dtype=np.int32),
        "sent_word_ptr": np.asarray(sent_word_ptr, dtype=np.int32),
        "words": np.asarray(words, dtype=np.int32),
        "vocab_size": len(word2id),
        "num_users": num_users,
        "num_items": num_items,
        "word2id": word2id,
    }
