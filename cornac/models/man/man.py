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
"""PyTorch network for MAN.

Implements the main network of

    Yang, P., Yin, M., Cao, H., et al. (2023).
    MAN: Main-auxiliary network with attentive interactions for review-based
    recommendation. Applied Intelligence (Springer).

The paper's model has a main network (Review Aspect Shift -> Interaction-based
Feature Learning -> ID embedding -> NFM rating head) and an auxiliary network
whose only role is to assist training of the main network's hidden features. The
auxiliary network is not needed to produce a rating (the iRev benchmark likewise
discards it), so only the main network is implemented here, following the
equations in section 4.2 of the paper.

Note: the corresponding module in the iRev benchmark does not run (its stacked
convolutions produce incompatible tensor shapes). This is a corrected,
length-preserving re-implementation faithful to the paper's architecture.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NFM(nn.Module):
    """Neural Factorization Machine rating head (ported from iRev framework)."""

    def __init__(self, dim, hidden=16):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
        self.fm_V = nn.Parameter(torch.randn(hidden, dim))
        self.mlp = nn.Linear(hidden, hidden)
        self.h = nn.Linear(hidden, 1, bias=False)
        self.dropout = nn.Dropout(0.5)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0.1)
        nn.init.uniform_(self.fm_V, -0.1, 0.1)
        nn.init.uniform_(self.h.weight, -0.1, 0.1)

    def forward(self, x):
        fm_linear = self.fc(x)
        inter_1 = torch.mm(x, self.fm_V.t()).pow(2)
        inter_2 = torch.mm(x.pow(2), self.fm_V.pow(2).t())
        bilinear = 0.5 * (inter_1 - inter_2)
        out = F.relu(self.mlp(bilinear))
        out = self.dropout(out)
        return self.h(out) + fm_linear


class MANModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        vocab_size,
        word_dim=300,
        id_embedding_size=32,
        n_filters=100,
        kernel_size=3,
        fc_dim=32,
        dropout_rate=0.5,
        eps=1.0,
        pretrained_word_embeddings=None,
    ):
        super().__init__()
        self.eps = eps
        pad = (kernel_size - 1) // 2

        self.user_word_embs = nn.Embedding(vocab_size, word_dim)
        self.item_word_embs = nn.Embedding(vocab_size, word_dim)

        # Review Aspect Shift: per-position aspect features (Cu, Ci).
        self.user_aspect_cnn = nn.Conv1d(word_dim, n_filters, kernel_size, padding=pad)
        self.item_aspect_cnn = nn.Conv1d(word_dim, n_filters, kernel_size, padding=pad)

        # Interaction-based Feature Learning: CNN + MLP over the shifted reviews.
        self.user_ifl_cnn = nn.Conv1d(word_dim, n_filters, kernel_size, padding=pad)
        self.item_ifl_cnn = nn.Conv1d(word_dim, n_filters, kernel_size, padding=pad)
        self.user_mlp = nn.Sequential(
            nn.Linear(n_filters, fc_dim), nn.ReLU(), nn.Linear(fc_dim, fc_dim), nn.ReLU()
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(n_filters, fc_dim), nn.ReLU(), nn.Linear(fc_dim, fc_dim), nn.ReLU()
        )
        self.interaction_mlp = nn.Sequential(
            nn.Linear(2 * fc_dim, fc_dim), nn.ReLU(), nn.Linear(fc_dim, fc_dim), nn.ReLU()
        )

        self.uid_embedding = nn.Embedding(num_users + 2, id_embedding_size)
        self.iid_embedding = nn.Embedding(num_items + 2, id_embedding_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.predict = NFM(fc_dim + 2 * id_embedding_size)

        self._reset_parameters()
        self._init_word_embeddings(pretrained_word_embeddings)

    def _reset_parameters(self):
        for cnn in [
            self.user_aspect_cnn,
            self.item_aspect_cnn,
            self.user_ifl_cnn,
            self.item_ifl_cnn,
        ]:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)
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

    def _aspect_shift(self, emb, aspects, other_prominent):
        """Reweight each word by the relevance of its aspect features to the
        other side's most prominent aspect (multi-representation attention)."""
        # Euclidean correlation score: frs = 1 / (eps + ||Cu,i - Ci||).
        dist = (aspects - other_prominent.unsqueeze(1)).pow(2).sum(-1).sqrt()
        score = 1.0 / (self.eps + dist)  # (bs, l)
        att = F.softmax(score, dim=1)
        return emb * att.unsqueeze(2)

    def _ifl(self, emb, ifl_cnn, mlp):
        fea = F.relu(ifl_cnn(emb.transpose(1, 2)))  # (bs, f, l)
        fea = fea.mean(dim=2)  # (bs, f)
        return mlp(fea)  # (bs, fc_dim)

    def forward(self, uids, iids, user_doc, item_doc):
        user_emb = self.user_word_embs(user_doc)  # (bs, l, d)
        item_emb = self.item_word_embs(item_doc)

        # Prominent aspect of each side (max over word positions).
        user_aspects = self.user_aspect_cnn(user_emb.transpose(1, 2)).transpose(1, 2)
        item_aspects = self.item_aspect_cnn(item_emb.transpose(1, 2)).transpose(1, 2)
        user_prominent = user_aspects.max(dim=1).values  # (bs, f)
        item_prominent = item_aspects.max(dim=1).values

        # Review Aspect Shift: shift user reviews toward item's prominent aspect
        # (and vice-versa).
        user_shifted = self._aspect_shift(user_emb, user_aspects, item_prominent)
        item_shifted = self._aspect_shift(item_emb, item_aspects, user_prominent)

        # Interaction-based Feature Learning.
        f_u = self._ifl(user_shifted, self.user_ifl_cnn, self.user_mlp)
        f_i = self._ifl(item_shifted, self.item_ifl_cnn, self.item_mlp)
        z_ui = self.interaction_mlp(torch.cat([f_u, f_i], dim=1))  # (bs, fc_dim)

        p_u = self.uid_embedding(uids)
        q_i = self.iid_embedding(iids)

        # Predictive hidden features -> NFM rating head.
        o = torch.cat([z_ui, p_u, q_i], dim=1)
        o = self.dropout(o)
        return self.predict(o).squeeze(1)
