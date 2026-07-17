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
"""PyTorch network for CARP (faithful port of the authors' official code).

    Li, C., Quan, C., Peng, L., Qi, Y., Deng, Y., & Wu, L. (2019).
    A Capsule Network for Recommendation and Explaining What You Like and
    Dislike. SIGIR 2019.

This is a faithful re-implementation of the official TensorFlow-1.x reference
(``CARP/runner/Carp_runner.py``). The full pipeline is reproduced:

    embedding + word mask
      -> gated-CNN viewpoint/aspect extraction (M aspects, Eq. 1)
      -> viewpoint self-attention (capsule routing, itr_0 iterations)
      -> M^2 logic units  g = [v_u (*) a_i ; v_u - a_i]
      -> sentiment capsules (positive/negative) via Routing-by-Bi-Agreement
      -> capsule lengths (used for the margin/sentiment loss)
      -> one-layer highway + rescaled-sigmoid rating + user/item bias

``forward`` returns ``(predict_rating, caps_len)``: the trainer needs the two
sentiment-capsule lengths for the margin loss, while ``score`` only consumes the
predicted rating.

Notes on faithfulness:
* Dropout is applied to weight matrices (DropConnect), exactly as the official
  ``tf.nn.dropout(W, keep_prob)`` calls. ``dropout_keep_prob`` is the KEEP
  probability (paper default 0.9), matching the TF semantics including the 1/keep
  up-scaling that ``F.dropout`` also performs.
* Capsule index 0 is the positive sentiment, index 1 the negative one.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_EPSILON = 1e-9


def _trunc_normal_(t, std):
    """Truncated normal clipped at +/-2*std, matching TF ``truncated_normal(stddev=std)``.

    ``nn.init.trunc_normal_`` interprets ``a``/``b`` as absolute cutoffs (default
    +/-2), so passing them as +/-2*std reproduces TF's 2-sigma truncation.
    """
    return nn.init.trunc_normal_(t, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


def squash(vector, dim=-2):
    """Capsule squashing non-linearity (official ``squash``, Eq. 5 in paper).

    Squashes along ``dim`` (the vector-length axis), preserving orientation while
    mapping the length into (0, 1).
    """
    vec_squared_norm = torch.sum(vector ** 2, dim=dim, keepdim=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / torch.sqrt(
        vec_squared_norm + _EPSILON
    )
    return scalar_factor * vector


def rescale_sigmoid(point, low_bound, up_bound):
    """Paper's ``f_C``: low + sigmoid(point) * (up - low) (official ``rescale_sigmoid``)."""
    return low_bound + torch.sigmoid(point) * (up_bound - low_bound)


class GatedConvAspect(nn.Module):
    """Conv encoder + per-aspect viewpoint gating and projection.

    Mirrors ``conv_layer`` + ``asps_gate`` + ``asps_prj`` of the official code for
    one side (user or item). Produces, for each of ``num_aspect`` aspects, a
    per-word projected representation ``(bs, L, num_filters)`` that the viewpoint
    self-attention then collapses into a single viewpoint vector.
    """

    def __init__(self, word_dim, num_filters, window_size, num_aspect):
        super().__init__()
        self.num_filters = num_filters
        self.num_aspect = num_aspect
        pad = (window_size - 1) // 2  # SAME padding for odd window (word length preserved)

        # Conv over words: kernel (window_size, word_dim), stride (1, word_dim).
        self.conv_w = nn.Parameter(torch.empty(num_filters, 1, window_size, word_dim))
        self.conv_b = nn.Parameter(torch.full((num_filters,), 0.1))
        self._conv_pad = (pad, 0)

        # Per-aspect learnable viewpoint/aspect embeddings q_x  (num_aspect, num_filters).
        self.aspect_embeds = nn.Parameter(torch.empty(num_aspect, num_filters))
        # Gate weights W_word / W_asps and bias, one per aspect.
        self.W_word = nn.Parameter(torch.empty(num_aspect, num_filters, num_filters))
        self.W_asps = nn.Parameter(torch.empty(num_aspect, num_filters, num_filters))
        self.b_gate = nn.Parameter(torch.full((num_aspect, num_filters), 0.1))
        # Projection weight W_prj, one per aspect.
        self.W_prj = nn.Parameter(torch.empty(num_aspect, num_filters, num_filters))

        self.reset_parameters()

    def reset_parameters(self):
        _trunc_normal_(self.conv_w, 0.3)
        _trunc_normal_(self.aspect_embeds, 0.3)
        _trunc_normal_(self.W_word, 0.1)
        _trunc_normal_(self.W_asps, 0.1)
        _trunc_normal_(self.W_prj, 0.3)

    def forward(self, emb, keep_prob):
        """emb: (bs, L, word_dim) already masked. Returns list of M (bs, L, f)."""
        p = 1.0 - keep_prob
        # --- Conv layer (DropConnect on conv weights) ---
        conv_w = F.dropout(self.conv_w, p=p, training=self.training)
        x = emb.unsqueeze(1)  # (bs, 1, L, word_dim)
        conv = F.conv2d(x, conv_w, bias=self.conv_b, stride=(1, emb.size(-1)),
                        padding=self._conv_pad)  # (bs, f, L, 1)
        contextual = F.relu(conv).squeeze(-1).transpose(1, 2)  # (bs, L, f)

        bs, L, f = contextual.shape
        projected = []
        for a in range(self.num_aspect):
            W_word = F.dropout(self.W_word[a], p=p, training=self.training)
            W_asps = F.dropout(self.W_asps[a], p=p, training=self.training)
            # Gate: sigmoid(c . W_word + q_x . W_asps + b)  ->  (bs, L, f)
            gate = torch.sigmoid(
                contextual.reshape(-1, f) @ W_word
                + self.aspect_embeds[a:a + 1] @ W_asps
                + self.b_gate[a]
            ).reshape(bs, L, f)
            gated = contextual * gate
            # Projection p = W_prj . s  (no dropout on W_prj in official asps_prj).
            proj = (gated.reshape(-1, f) @ self.W_prj[a]).reshape(bs, L, f)
            projected.append(proj)
        return projected


def viewpoint_self_attn(proj_words, mask, prj_bias, n_iter):
    """Collapse per-word aspect features into one viewpoint via capsule routing.

    Faithful port of ``simple_self_attn``: an ``n_iter``-iteration dynamic routing
    (with ``squash``) from the ``L`` projected words to a single output capsule of
    dimension ``num_filters``. Padded words are masked out of the routing logits.

    proj_words: (bs, L, f)   mask: (bs, L)   prj_bias: (1, 1, f)
    Returns: (bs, 1, f)
    """
    bs, L, f = proj_words.shape
    u_hat = proj_words.unsqueeze(-1)  # (bs, L, f, 1)
    u_hat_stopped = u_hat.detach()

    b_ij = torch.zeros(bs, L, 1, 1, device=proj_words.device, dtype=proj_words.dtype)
    mask_e = mask.unsqueeze(-1).unsqueeze(-1)  # (bs, L, 1, 1)

    v_j = None
    for r in range(n_iter):
        if r > 0:
            b_ij = b_ij * mask_e
        c_ij = torch.softmax(b_ij, dim=1)  # over words
        cur = u_hat if r == n_iter - 1 else u_hat_stopped
        s_j = torch.sum(c_ij * cur, dim=1, keepdim=True) + prj_bias.unsqueeze(-1)
        v_j = squash(s_j, dim=-2)  # (bs, 1, f, 1)
        if r < n_iter - 1:
            u_produce_v = torch.sum(u_hat_stopped * v_j, dim=2, keepdim=True)  # (bs, L, 1, 1)
            b_ij = b_ij + u_produce_v
    return v_j.squeeze(-1)  # (bs, 1, f)


class SentimentCapsule(nn.Module):
    """Positive/negative sentiment capsules with Routing-by-Bi-Agreement (RBiA).

    Faithful port of ``caps_layer_2``: transforms each of the ``num_logic`` logic
    units into votes for two output capsules of dimension ``latent_dim`` and routes
    them with the bi-agreement coupling ``c = sqrt(softmax_intra * softmax_inter)``.
    """

    def __init__(self, num_logic, logic_dim, latent_dim, n_iter):
        super().__init__()
        self.num_logic = num_logic
        self.latent_dim = latent_dim
        self.n_caps = 2  # positive / negative
        self.n_iter = n_iter
        # caps_W: (1, num_logic, n_caps * latent_dim, logic_dim, 1)
        self.caps_W = nn.Parameter(
            torch.empty(1, num_logic, self.n_caps * latent_dim, logic_dim, 1)
        )
        # caps_b: (1, 1, n_caps, latent_dim, 1)
        self.caps_b = nn.Parameter(torch.full((1, 1, self.n_caps, latent_dim, 1), 0.1))
        _trunc_normal_(self.caps_W, 0.3)

    def forward(self, logic_units):
        """logic_units: (bs, num_logic, logic_dim). Returns v_sent: (bs, n_caps, latent_dim)."""
        bs = logic_units.size(0)
        # u_hat = sum over logic_dim of caps_W * logic_unit  -> votes.
        x = logic_units.unsqueeze(2).unsqueeze(-1)  # (bs, num_logic, 1, logic_dim, 1)
        u_hat = torch.sum(self.caps_W * x, dim=3, keepdim=True)  # (bs, num_logic, n_caps*k, 1, 1)
        u_hat = u_hat.reshape(bs, self.num_logic, self.n_caps, self.latent_dim, 1)
        u_hat_stopped = u_hat.detach()

        b_ij = torch.zeros(bs, self.num_logic, self.n_caps, 1, 1,
                           device=logic_units.device, dtype=logic_units.dtype)

        v_j = None
        for r in range(self.n_iter):
            c_intra = torch.softmax(b_ij, dim=1)  # over logic units
            c_inter = torch.softmax(b_ij, dim=2)  # over the two sentiments
            c_mul = torch.sqrt(c_intra * c_inter + _EPSILON)
            c_sum = torch.sum(c_mul, dim=1, keepdim=True)
            c_ij = c_mul / (c_sum + _EPSILON)

            cur = u_hat if r == self.n_iter - 1 else u_hat_stopped
            s_j = torch.sum(c_ij * cur, dim=1, keepdim=True) + self.caps_b
            v_j = squash(s_j, dim=-2)  # (bs, 1, n_caps, k, 1)
            if r < self.n_iter - 1:
                u_produce_v = torch.sum(u_hat_stopped * v_j, dim=3, keepdim=True)
                b_ij = b_ij + u_produce_v
        return v_j.squeeze(1).squeeze(-1)  # (bs, n_caps, latent_dim)


class Highway(nn.Module):
    """One-layer highway network (official ``highway`` with the gate on)."""

    def __init__(self, dim):
        super().__init__()
        self.W_trans = nn.Parameter(torch.empty(dim, dim))
        self.b_trans = nn.Parameter(torch.full((dim,), 0.1))
        self.W_gate = nn.Parameter(torch.empty(dim, dim))
        self.b_gate = nn.Parameter(torch.full((dim,), 0.1))
        _trunc_normal_(self.W_trans, 0.3)
        _trunc_normal_(self.W_gate, 0.3)

    def forward(self, x, keep_prob):
        p = 1.0 - keep_prob
        W_trans = F.dropout(self.W_trans, p=p, training=self.training)
        high = torch.tanh(x @ W_trans + self.b_trans)
        W_gate = F.dropout(self.W_gate, p=p, training=self.training)
        gate = torch.sigmoid(x @ W_gate + self.b_gate)
        return high * gate + x * (1 - gate)


class CARPModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        vocab_size,
        word_dim=300,
        latent_dim=25,
        n_filters=50,
        num_aspect=5,
        kernel_size=3,
        itr_self_attn=2,
        itr_routing=3,
        min_rating=1.0,
        max_rating=5.0,
        dropout_keep_prob=0.9,
        pretrained_word_embeddings=None,
    ):
        super().__init__()
        self.num_aspect = num_aspect
        self.n_filters = n_filters
        self.latent_dim = latent_dim
        self.itr_self_attn = itr_self_attn
        self.min_rating = float(min_rating)
        self.max_rating = float(max_rating)
        self.dropout_keep_prob = dropout_keep_prob

        # Word embeddings with an extra zero padding row at index ``vocab_size``.
        self.user_word_embs = nn.Embedding(vocab_size + 1, word_dim, padding_idx=vocab_size)
        self.item_word_embs = nn.Embedding(vocab_size + 1, word_dim, padding_idx=vocab_size)
        self.vocab_size = vocab_size

        # Gated-CNN viewpoint/aspect extractors (separate user/item).
        self.user_encoder = GatedConvAspect(word_dim, n_filters, kernel_size, num_aspect)
        self.item_encoder = GatedConvAspect(word_dim, n_filters, kernel_size, num_aspect)

        # Per-aspect projection biases for the viewpoint self-attention.
        self.user_prj_bias = nn.Parameter(torch.full((num_aspect, 1, 1, n_filters), 0.1))
        self.item_prj_bias = nn.Parameter(torch.full((num_aspect, 1, 1, n_filters), 0.1))

        # Sentiment capsules over the M^2 logic units (each logic unit is 2*f dim).
        self.sentiment = SentimentCapsule(
            num_logic=num_aspect * num_aspect,
            logic_dim=2 * n_filters,
            latent_dim=latent_dim,
            n_iter=itr_routing,
        )

        # One-layer highway per sentiment + degree heads.
        self.highway_pos = Highway(latent_dim)
        self.highway_neg = Highway(latent_dim)
        self.W_pos = nn.Parameter(torch.empty(latent_dim, 1))
        self.W_neg = nn.Parameter(torch.empty(latent_dim, 1))
        self.b_pos = nn.Parameter(torch.full((1,), 3.0))
        self.b_neg = nn.Parameter(torch.full((1,), 1.0))
        _trunc_normal_(self.W_pos, 0.3)
        _trunc_normal_(self.W_neg, 0.3)

        # Interaction bias terms.
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.normal_(self.user_bias.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.item_bias.weight, mean=0.0, std=0.02)

        self._init_word_embeddings(pretrained_word_embeddings)

    def _init_word_embeddings(self, pretrained):
        if pretrained is not None:
            w2v = torch.from_numpy(np.asarray(pretrained, dtype="float32"))
            self.user_word_embs.weight.data[: self.vocab_size].copy_(w2v)
            self.item_word_embs.weight.data[: self.vocab_size].copy_(w2v)
        else:
            # Official runner random-inits ALL words as uniform [0, 1) (np.random.rand);
            # an all-positive init, but faithful to the reference code.
            nn.init.uniform_(self.user_word_embs.weight, 0.0, 1.0)
            nn.init.uniform_(self.item_word_embs.weight, 0.0, 1.0)
        # Keep the padding row at zero.
        with torch.no_grad():
            self.user_word_embs.weight[self.vocab_size].zero_()
            self.item_word_embs.weight[self.vocab_size].zero_()

    def _side(self, word_embs, encoder, prj_bias, doc, mask):
        keep = self.dropout_keep_prob
        emb = word_embs(doc) * mask.unsqueeze(-1)  # (bs, L, d), padded words zeroed
        projected = encoder(emb, keep)  # list of M (bs, L, f)
        viewpoints = []
        for a in range(self.num_aspect):
            v = viewpoint_self_attn(
                projected[a], mask, prj_bias[a], self.itr_self_attn
            )  # (bs, 1, f)
            viewpoints.append(v)
        return viewpoints  # list of M (bs, 1, f)

    def forward(self, uids, iids, user_doc, item_doc, user_mask, item_mask):
        user_vps = self._side(self.user_word_embs, self.user_encoder,
                              self.user_prj_bias, user_doc, user_mask)
        item_vps = self._side(self.item_word_embs, self.item_encoder,
                              self.item_prj_bias, item_doc, item_mask)

        # Logic units: for every user-viewpoint x item-aspect pair -> [u*i ; u-i].
        logic = []
        for u in user_vps:            # (bs, 1, f)
            for it in item_vps:       # (bs, 1, f)
                logic.append(torch.cat([u * it, u - it], dim=2))  # (bs, 1, 2f)
        logic_units = torch.cat(logic, dim=1)  # (bs, M^2, 2f)

        # Sentiment capsules via Routing-by-Bi-Agreement.
        v_sent = self.sentiment(logic_units)  # (bs, 2, k)
        caps_len = torch.sqrt(torch.sum(v_sent ** 2, dim=-1) + _EPSILON)  # (bs, 2)

        v_pos = v_sent[:, 0, :]  # (bs, k)
        v_neg = v_sent[:, 1, :]
        pos_len = caps_len[:, 0:1]  # (bs, 1)
        neg_len = caps_len[:, 1:2]

        keep = self.dropout_keep_prob
        abs_pos = self.highway_pos(v_pos, keep)
        abs_neg = self.highway_neg(v_neg, keep)

        W_pos = F.dropout(self.W_pos, p=1.0 - keep, training=self.training)
        W_neg = F.dropout(self.W_neg, p=1.0 - keep, training=self.training)
        pos_rate = abs_pos @ W_pos + self.b_pos  # (bs, 1)
        neg_rate = abs_neg @ W_neg + self.b_neg

        u_b = self.user_bias(uids)  # (bs, 1)
        i_b = self.item_bias(iids)
        rating = rescale_sigmoid(
            pos_len * pos_rate - neg_len * neg_rate, self.min_rating, self.max_rating
        ) + u_b + i_b
        return rating.squeeze(1), caps_len
