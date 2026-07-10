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
"""PyTorch network for DAML.

Ported from the iRev benchmark implementation of

    Liu, D., Li, J., Du, B., Chang, J., & Gao, R. (2019).
    DAML: Dual Attention Mutual Learning between Ratings and Reviews for
    Item Recommendation. KDD 2019.

The iRev module returned two stacked (document + id) features per user/item that
a shared FusionLayer concatenated before an LFM head. Those steps are folded in
here so the module maps ``(uids, iids, user_doc, item_doc)`` to a scalar rating.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LFM(nn.Module):
    """Latent Factor Model rating head (ported from iRev framework)."""

    def __init__(self, dim, num_users, num_items):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
        self.b_users = nn.Parameter(torch.randn(num_users, 1))
        self.b_items = nn.Parameter(torch.randn(num_items, 1))
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc.bias, 0.5, 1.5)
        nn.init.uniform_(self.b_users, 0.5, 1.5)
        nn.init.uniform_(self.b_items, 0.5, 1.5)

    def forward(self, feature, user_id, item_id):
        return self.fc(feature) + self.b_users[user_id] + self.b_items[item_id]


class DAMLModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        vocab_size,
        word_dim=300,
        id_embedding_size=32,
        n_filters=100,
        kernel_size=3,
        dropout_rate=0.5,
        pretrained_word_embeddings=None,
    ):
        super().__init__()
        self.n_filters = n_filters

        self.user_word_embs = nn.Embedding(vocab_size, word_dim)
        self.item_word_embs = nn.Embedding(vocab_size, word_dim)

        self.word_cnn = nn.Conv2d(1, 1, (5, word_dim), padding=(2, 0))
        self.user_doc_cnn = nn.Conv2d(1, n_filters, (kernel_size, word_dim), padding=(1, 0))
        self.item_doc_cnn = nn.Conv2d(1, n_filters, (kernel_size, word_dim), padding=(1, 0))
        self.user_abs_cnn = nn.Conv2d(1, n_filters, (kernel_size, n_filters))
        self.item_abs_cnn = nn.Conv2d(1, n_filters, (kernel_size, n_filters))

        self.unfold = nn.Unfold((3, n_filters), padding=(1, 0))

        self.user_fc = nn.Linear(n_filters, id_embedding_size)
        self.item_fc = nn.Linear(n_filters, id_embedding_size)

        # +2 mirrors iRev (padding indices for cold user/item slots).
        self.uid_embedding = nn.Embedding(num_users + 2, id_embedding_size)
        self.iid_embedding = nn.Embedding(num_items + 2, id_embedding_size)

        self.dropout = nn.Dropout(dropout_rate)
        # num_fea=2 features, fused by concatenation for user & item -> 4 * id.
        self.predict = LFM(id_embedding_size * 4, num_users, num_items)

        self._reset_parameters()
        self._init_word_embeddings(pretrained_word_embeddings)

    def _reset_parameters(self):
        for cnn in [
            self.word_cnn,
            self.user_doc_cnn,
            self.item_doc_cnn,
            self.user_abs_cnn,
            self.item_abs_cnn,
        ]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.uniform_(cnn.bias, -0.1, 0.1)
        for fc in [self.user_fc, self.item_fc]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            nn.init.constant_(fc.bias, 0.1)
        nn.init.uniform_(self.uid_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.iid_embedding.weight, -0.1, 0.1)

    def _init_word_embeddings(self, pretrained):
        if pretrained is not None:
            w2v = torch.from_numpy(np.asarray(pretrained, dtype="float32"))
            self.user_word_embs.weight.data.copy_(w2v)
            self.item_word_embs.weight.data.copy_(w2v)
        else:
            nn.init.uniform_(self.user_word_embs.weight, -0.1, 0.1)
            nn.init.uniform_(self.item_word_embs.weight, -0.1, 0.1)

    def local_attention_cnn(self, word_embs, doc_cnn):
        local_att_words = self.word_cnn(word_embs.unsqueeze(1))
        local_word_weight = torch.sigmoid(local_att_words.squeeze(1))
        word_embs = word_embs * local_word_weight
        return doc_cnn(word_embs.unsqueeze(1))

    def local_pooling_cnn(self, feature, attention, cnn, fc):
        bs, n_filters, doc_len, _ = feature.shape
        feature = feature.permute(0, 3, 2, 1)  # bs * 1 * doc_len * n_filters
        attention = attention.reshape(bs, 1, doc_len, 1)
        pools = feature * attention
        pools = self.unfold(pools)
        pools = pools.reshape(bs, 3, n_filters, doc_len)
        pools = pools.sum(dim=1, keepdims=True)
        pools = pools.transpose(2, 3)  # bs * 1 * doc_len * n_filters

        abs_fea = cnn(pools).squeeze(3)
        abs_fea = F.avg_pool1d(abs_fea, abs_fea.size(2))
        abs_fea = F.relu(fc(abs_fea.squeeze(2)))
        return abs_fea

    def forward(self, uids, iids, user_doc, item_doc):
        user_word_embs = self.user_word_embs(user_doc)
        item_word_embs = self.item_word_embs(item_doc)

        user_local_fea = self.local_attention_cnn(user_word_embs, self.user_doc_cnn)
        item_local_fea = self.local_attention_cnn(item_word_embs, self.item_doc_cnn)

        euclidean = (user_local_fea - item_local_fea.permute(0, 1, 3, 2)).pow(2).sum(1).sqrt()
        attention_matrix = 1.0 / (1 + euclidean)
        user_attention = attention_matrix.sum(2)
        item_attention = attention_matrix.sum(1)

        user_doc_fea = self.local_pooling_cnn(
            user_local_fea, user_attention, self.user_abs_cnn, self.user_fc
        )
        item_doc_fea = self.local_pooling_cnn(
            item_local_fea, item_attention, self.item_abs_cnn, self.item_fc
        )

        uid_emb = self.uid_embedding(uids)
        iid_emb = self.iid_embedding(iids)

        # num_fea=2: stack doc feature with id embedding, then FusionLayer(cat).
        user_fea = torch.cat([user_doc_fea, uid_emb], dim=1)  # (bs, 2*id)
        item_fea = torch.cat([item_doc_fea, iid_emb], dim=1)  # (bs, 2*id)
        fused = torch.cat([user_fea, item_fea], dim=1)  # (bs, 4*id)
        fused = self.dropout(fused)
        return self.predict(fused, uids, iids).squeeze(1)
