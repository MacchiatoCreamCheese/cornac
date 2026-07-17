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
"""PyTorch networks for MAN (faithful, implemented from the paper).

    Yang, P., Xiao, Y., Zheng, W., Jiao, X., Zhu, K., Sun, C., & Liu, L. (2023).
    MAN: Main-auxiliary network with attentive interactions for review-based
    recommendation. Applied Intelligence 53:12955-12970.

No official repository exists; this implements the paper's equations directly:

* ``MainFeatureNet`` (Eq. 1-16): Review Aspect Shift (RAS) over RO/ORT ->
  Interaction-based Feature Learning (IFL) -> ID embeddings (IDE1) ->
  predictive hidden features ``O = [Z_ui, P_u, Q_i]``.
* ``AuxFeatureNet`` (Eq. 17-28): word-based attention -> multi-head
  self-attention -> CNN + MLP over RT -> ID embeddings (IDE2) ->
  accurate hidden features ``O* = [Z*_ui, P*_u, Q*_i]``.
* ``NFM`` (Eq. 29-30): rating head; the model holds two independent instances
  (NFM1 on O, NFM2 on O*).

Training (Algorithm 1) is a 3-step alternating scheme driven by the recommender:
J_A trains the auxiliary net, J_B distills O toward O*, J_C trains NFM1.
Testing (Algorithm 2) uses the main network only.

Documented interpretations (paper ambiguities):
* The item's "most prominent aspect feature" C_i is the max over word positions
  of its aspect features (the text's "only one convolution filter" reading gives
  incompatible shapes for ||C_u,i - C_i||).
* E_u is mean-pooled over word positions before the 2-layer MLP (the paper's
  W_{u,c} in R^{f mu x gamma} treats the doc stream as mu=1 review).
* Word embeddings are held fixed during training (paper Eq. 1: "held fixed
  throughout the training process").
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _trunc_normal_(t, std=0.1):
    """Truncated normal (mean 0, +-2 std), matching the paper's init (Sec. 5.4)."""
    return nn.init.trunc_normal_(t, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


class NFM(nn.Module):
    """Neural Factorization Machine rating head (Eq. 29-30); 2-layer MLP (Fig. 7)."""

    def __init__(self, dim, hidden=16, dropout=0.5):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
        self.fm_V = nn.Parameter(torch.randn(hidden, dim))
        self.mlp = nn.Linear(hidden, hidden)
        self.h = nn.Linear(hidden, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        _trunc_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.1)
        _trunc_normal_(self.fm_V)
        _trunc_normal_(self.mlp.weight)
        nn.init.constant_(self.mlp.bias, 0.1)
        _trunc_normal_(self.h.weight)

    def forward(self, x):
        fm_linear = self.fc(x)
        inter_1 = torch.mm(x, self.fm_V.t()).pow(2)
        inter_2 = torch.mm(x.pow(2), self.fm_V.pow(2).t())
        bilinear = 0.5 * (inter_1 - inter_2)
        out = F.relu(self.mlp(bilinear))
        out = self.dropout(out)
        return self.h(out) + fm_linear


class _TwoLayerMLP(nn.Module):
    """delta(W2 delta(W1 x + b1) + b2) -- the paper's 2-layer MLPs (Fig. 3)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, out_dim)
        self.l2 = nn.Linear(out_dim, out_dim)
        for l in (self.l1, self.l2):
            _trunc_normal_(l.weight)
            nn.init.constant_(l.bias, 0.1)

    def forward(self, x):
        return F.relu(self.l2(F.relu(self.l1(x))))


class MainFeatureNet(nn.Module):
    """RAS + IFL + IDE1 -> predictive hidden features O (Eq. 1-16)."""

    def __init__(self, num_users, num_items, word_embs, word_dim,
                 id_embedding_size, n_filters, kernel_size, fc_dim, eps=1.0):
        super().__init__()
        self.word_embs = word_embs  # shared frozen embedding table
        self.eps = eps
        pad = (kernel_size - 1) // 2

        # RAS aspect CNNs (Eq. 2-4), separate for user (RO) and item (ORT) streams.
        self.user_aspect_cnn = nn.Conv1d(word_dim, n_filters, kernel_size, padding=pad)
        self.item_aspect_cnn = nn.Conv1d(word_dim, n_filters, kernel_size, padding=pad)

        # IFL CNNs (Eq. 9) + 2-layer MLPs (Eq. 11) + interaction MLP (Eq. 13).
        self.user_ifl_cnn = nn.Conv1d(word_dim, n_filters, kernel_size, padding=pad)
        self.item_ifl_cnn = nn.Conv1d(word_dim, n_filters, kernel_size, padding=pad)
        self.user_mlp = _TwoLayerMLP(n_filters, fc_dim)
        self.item_mlp = _TwoLayerMLP(n_filters, fc_dim)
        self.interaction_mlp = _TwoLayerMLP(2 * fc_dim, fc_dim)

        # IDE1 (Eq. 14-15); paper: uniform(-1, 1) init.
        self.uid_embedding = nn.Embedding(num_users, id_embedding_size)
        self.iid_embedding = nn.Embedding(num_items, id_embedding_size)
        nn.init.uniform_(self.uid_embedding.weight, -1.0, 1.0)
        nn.init.uniform_(self.iid_embedding.weight, -1.0, 1.0)

        for cnn in (self.user_aspect_cnn, self.item_aspect_cnn,
                    self.user_ifl_cnn, self.item_ifl_cnn):
            _trunc_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

    def _aspect_shift(self, emb, aspects, other_prominent):
        """Multi-representation attention (Eq. 5-8): reweight words by relevance
        of their aspect features to the other side's most prominent aspect."""
        # +1e-8 inside the sqrt: the prominent aspect is the max over words, so at
        # the argmax word the distance is exactly 0 and sqrt'(0)=inf would NaN the
        # backward pass.
        dist = ((aspects - other_prominent.unsqueeze(1)).pow(2).sum(-1) + 1e-8).sqrt()
        score = 1.0 / (self.eps + dist)  # (bs, l)  Eq. 5-6
        att = F.softmax(score, dim=1)  # Eq. 7
        return emb * att.unsqueeze(2)  # Eq. 8

    def _ifl_side(self, emb, ifl_cnn, mlp):
        fea = F.relu(ifl_cnn(emb.transpose(1, 2)))  # (bs, f, l)  Eq. 9-10
        fea = fea.mean(dim=2)  # (bs, f)  pool over word positions
        return mlp(fea)  # (bs, fc_dim)  Eq. 11

    def forward(self, uids, iids, ro_doc, ort_doc):
        user_emb = self.word_embs(ro_doc)  # (bs, l, d)
        item_emb = self.word_embs(ort_doc)

        # Aspect features and each side's most prominent aspect (max over words).
        user_aspects = self.user_aspect_cnn(user_emb.transpose(1, 2)).transpose(1, 2)
        item_aspects = self.item_aspect_cnn(item_emb.transpose(1, 2)).transpose(1, 2)
        user_prominent = user_aspects.max(dim=1).values  # (bs, f)
        item_prominent = item_aspects.max(dim=1).values

        # RAS: shift each stream toward the other side's prominent aspect.
        user_shifted = self._aspect_shift(user_emb, user_aspects, item_prominent)
        item_shifted = self._aspect_shift(item_emb, item_aspects, user_prominent)

        # IFL.
        f_u = self._ifl_side(user_shifted, self.user_ifl_cnn, self.user_mlp)
        f_i = self._ifl_side(item_shifted, self.item_ifl_cnn, self.item_mlp)
        z_ui = self.interaction_mlp(torch.cat([f_u, f_i], dim=1))  # Eq. 12-13

        p_u = self.uid_embedding(uids)  # Eq. 14
        q_i = self.iid_embedding(iids)  # Eq. 15
        return torch.cat([z_ui, p_u, q_i], dim=1)  # O  (Eq. 16)


class AuxFeatureNet(nn.Module):
    """Word attention + multi-head self-attention + CNN/MLP + IDE2 -> O* (Eq. 17-28)."""

    def __init__(self, num_users, num_items, word_embs, word_dim,
                 id_embedding_size, n_filters, kernel_size, fc_dim,
                 att_hidden=64, n_heads=4, ff_dim=128, dropout=0.5):
        super().__init__()
        self.word_embs = word_embs
        pad = (kernel_size - 1) // 2

        # Word-based attention (Eq. 18-19).
        self.att_W = nn.Linear(word_dim, att_hidden)
        self.att_h = nn.Linear(att_hidden, 1)
        _trunc_normal_(self.att_W.weight)
        nn.init.constant_(self.att_W.bias, 0.1)
        _trunc_normal_(self.att_h.weight)
        nn.init.constant_(self.att_h.bias, 0.1)

        # Multi-head self-attention (Eq. 20-22): 1 encoder layer (paper Sec. 5.4).
        self.encoder = nn.TransformerEncoderLayer(
            d_model=word_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True,
        )

        # CNN + 2-layer MLP (Eq. 23-25).
        self.cnn = nn.Conv1d(word_dim, n_filters, kernel_size, padding=pad)
        _trunc_normal_(self.cnn.weight)
        nn.init.constant_(self.cnn.bias, 0.1)
        self.mlp = _TwoLayerMLP(n_filters, fc_dim)

        # IDE2 (Eq. 26-27) -- independent from IDE1.
        self.uid_embedding = nn.Embedding(num_users, id_embedding_size)
        self.iid_embedding = nn.Embedding(num_items, id_embedding_size)
        nn.init.uniform_(self.uid_embedding.weight, -1.0, 1.0)
        nn.init.uniform_(self.iid_embedding.weight, -1.0, 1.0)

    def forward(self, uids, iids, rt_doc):
        emb = self.word_embs(rt_doc)  # V**  (bs, psi, d)  Eq. 17

        # Word-based attention weights (Eq. 18), applied to the word vectors (Eq. 19).
        scores = self.att_h(torch.relu(self.att_W(emb))).squeeze(-1)  # (bs, psi)
        att = F.softmax(scores, dim=1)
        v = emb * att.unsqueeze(2)  # V*

        # Multi-head self-attention (Eq. 20-22).
        v = self.encoder(v)  # (bs, psi, d)

        # CNN + MLP (Eq. 23-25).
        fea = F.relu(self.cnn(v.transpose(1, 2)))  # (bs, f, psi)
        fea = fea.mean(dim=2)
        z_star = self.mlp(fea)  # (bs, fc_dim)

        p_u = self.uid_embedding(uids)
        q_i = self.iid_embedding(iids)
        return torch.cat([z_star, p_u, q_i], dim=1)  # O*  (Eq. 28)


class MANModel(nn.Module):
    """Container holding both networks and both NFM heads.

    The trainer drives Algorithm 1's three steps through the exposed submodules;
    ``forward`` implements Algorithm 2 (test-time main-network prediction).
    """

    def __init__(
        self,
        num_users,
        num_items,
        vocab_size,
        word_dim=64,
        id_embedding_size=32,
        n_filters=100,
        kernel_size=3,
        fc_dim=50,
        att_hidden=64,
        n_heads=4,
        ff_dim=128,
        dropout_rate=0.5,
        eps=1.0,
        pretrained_word_embeddings=None,
    ):
        super().__init__()
        # Shared word embedding table, held FIXED during training (paper Eq. 1).
        self.word_embs = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        if pretrained_word_embeddings is not None:
            w2v = torch.from_numpy(np.asarray(pretrained_word_embeddings, dtype="float32"))
            self.word_embs.weight.data.copy_(w2v)
        else:
            nn.init.uniform_(self.word_embs.weight, -0.1, 0.1)
        with torch.no_grad():
            self.word_embs.weight[0].zero_()
        self.word_embs.weight.requires_grad = False

        self.main_net = MainFeatureNet(
            num_users, num_items, self.word_embs, word_dim,
            id_embedding_size, n_filters, kernel_size, fc_dim, eps=eps,
        )
        self.aux_net = AuxFeatureNet(
            num_users, num_items, self.word_embs, word_dim,
            id_embedding_size, n_filters, kernel_size, fc_dim,
            att_hidden=att_hidden, n_heads=n_heads, ff_dim=ff_dim,
            dropout=dropout_rate,
        )
        o_dim = fc_dim + 2 * id_embedding_size
        self.nfm1 = NFM(o_dim, dropout=dropout_rate)  # main head  (r-hat from O)
        self.nfm2 = NFM(o_dim, dropout=dropout_rate)  # aux head   (r-hat* from O*)

    def forward(self, uids, iids, ro_doc, ort_doc):
        """Algorithm 2: prediction with the main network only."""
        o = self.main_net(uids, iids, ro_doc, ort_doc)
        return self.nfm1(o).squeeze(1)
