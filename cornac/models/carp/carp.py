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
"""PyTorch network for CARP.

Ported from the iRev benchmark implementation of

    Li, C., Quan, C., Peng, L., Qi, Y., Deng, Y., & Wu, L. (2019).
    A Capsule Network for Recommendation and Explaining What You Like and Dislike.
    SIGIR 2019.

The original iRev module returned raw user/item features that were combined by
a shared FusionLayer + LFM prediction head. Here those pieces are folded into a
single ``nn.Module`` that maps ``(uids, iids, user_doc, item_doc)`` to a scalar
rating, so the module is self-contained within Cornac.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewPoint(nn.Module):
    """Sigmoid-gated CNN over review words, modulated by the id embedding."""

    def __init__(self, n_filters, kernel_size, word_dim, id_embedding_size):
        super().__init__()
        self.cnn = nn.Conv2d(1, n_filters, (kernel_size, word_dim))
        self.review_linear = nn.Linear(n_filters, 1)
        self.id_linear = nn.Linear(id_embedding_size, n_filters, bias=False)
        self.global_viewpoint = nn.Linear(n_filters, id_embedding_size, bias=False)
        self.reset_parameters()

    def forward(self, fea, id_fea):
        fea = F.relu(self.cnn(fea.unsqueeze(1)).squeeze(3)).transpose(2, 1)
        viewpoint = torch.sigmoid(
            self.review_linear(fea) + self.id_linear(id_fea).unsqueeze(1)
        )
        context = fea * viewpoint
        return self.global_viewpoint(context)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.cnn.weight)
        nn.init.constant_(self.cnn.bias, 0.1)
        for x in [self.review_linear, self.id_linear, self.global_viewpoint]:
            nn.init.uniform_(x.weight, -0.1, 0.1)
            if x.bias is not None:
                nn.init.constant_(x.bias, 0.1)


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


class CARPModel(nn.Module):
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
        self.kernel_size = kernel_size

        self.user_word_embs = nn.Embedding(vocab_size, word_dim)
        self.item_word_embs = nn.Embedding(vocab_size, word_dim)

        self.id_user_emb = nn.Embedding(num_users, id_embedding_size)
        self.id_item_emb = nn.Embedding(num_items, id_embedding_size)

        self.user_viewpoint = ViewPoint(n_filters, kernel_size, word_dim, id_embedding_size)
        self.item_viewpoint = ViewPoint(n_filters, kernel_size, word_dim, id_embedding_size)

        self.logic_unit = nn.Linear(id_embedding_size * 2, id_embedding_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

        # FusionLayer(cat) over (u_fea, i_fea) -> id_embedding_size * 2, then LFM head.
        self.fusion = nn.Linear(id_embedding_size * 2, id_embedding_size * 2)
        self.predict = LFM(id_embedding_size * 2, num_users, num_items)

        self._reset_parameters()
        self._init_word_embeddings(pretrained_word_embeddings)

    def _reset_parameters(self):
        for emb in [self.id_user_emb, self.id_item_emb]:
            nn.init.uniform_(emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.logic_unit.weight, -0.1, 0.1)
        nn.init.uniform_(self.fusion.weight, -0.1, 0.1)
        nn.init.constant_(self.fusion.bias, 0.1)

    def _init_word_embeddings(self, pretrained):
        if pretrained is not None:
            w2v = torch.from_numpy(np.asarray(pretrained, dtype="float32"))
            self.user_word_embs.weight.data.copy_(w2v)
            self.item_word_embs.weight.data.copy_(w2v)
        else:
            nn.init.uniform_(self.user_word_embs.weight, -0.1, 0.1)
            nn.init.uniform_(self.item_word_embs.weight, -0.1, 0.1)

    def forward(self, uids, iids, user_doc, item_doc):
        user_doc = self.user_word_embs(user_doc)  # (bs, doc_len, word_dim)
        item_doc = self.item_word_embs(item_doc)

        user_ids = self.id_user_emb(uids)
        item_ids = self.id_item_emb(iids)

        pad_size = (self.kernel_size - 1) // 2
        u_fea = F.pad(user_doc, (0, 0, pad_size, pad_size), "constant", 0)
        i_fea = F.pad(item_doc, (0, 0, pad_size, pad_size), "constant", 0)

        u_viewpoints = self.user_viewpoint(u_fea, user_ids)
        i_viewpoints = self.item_viewpoint(i_fea, item_ids)

        u_att = torch.mean(u_viewpoints, dim=1)
        i_att = torch.mean(i_viewpoints, dim=1)
        u_att = F.softmax(u_viewpoints * u_att.unsqueeze(1), dim=1)
        i_att = F.softmax(i_viewpoints * i_att.unsqueeze(1), dim=1)

        u_fea = torch.sum(u_viewpoints * u_att, 1)  # (bs, id_embedding_size)
        i_fea = torch.sum(i_viewpoints * i_att, 1)

        # Capsule "logic unit" (sentiment interaction of the two viewpoints).
        gated_unit = torch.cat([(u_fea - i_fea), (u_fea * i_fea)], dim=1)
        gated_unit = self.logic_unit(gated_unit)
        coupling_coeff = F.softmax(gated_unit, dim=1)
        u_fea = u_fea * coupling_coeff
        i_fea = i_fea * coupling_coeff

        # FusionLayer(cat) + dropout + LFM head.
        fused = torch.cat([u_fea, i_fea], dim=1)
        fused = self.fusion(fused)
        fused = self.dropout(fused)
        return self.predict(fused, uids, iids).squeeze(1)
