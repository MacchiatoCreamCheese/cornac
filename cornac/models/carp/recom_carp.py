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

import copy

import numpy as np
from tqdm.auto import tqdm

from ..recommender import Recommender
from ...exception import ScoreException


class CARP(Recommender):
    """Capsule network for Rating Prediction (CARP).

    Faithful Cornac/PyTorch port of the authors' official implementation
    (https://github.com/WHUIR/CARP). For each user-item pair CARP extracts ``M``
    viewpoints from the user document and ``M`` aspects from the item document with
    a gated CNN, pairs them into ``M^2`` logic units, routes the logic units into a
    positive and a negative sentiment capsule via Routing-by-Bi-Agreement, and
    predicts the rating from the two sentiment magnitudes through a highway network
    and a rescaled sigmoid. Training is multi-task: a rating MSE plus a capsule
    margin (sentiment classification) loss.

    Parameters
    ----------
    name: string, default: 'CARP'
        The name of the recommender model.

    embedding_size: int, default: 300
        Word embedding size (``d`` in the paper).

    latent_dim: int, default: 25
        Sentiment-capsule dimension (``k`` in the paper).

    n_filters: int, default: 50
        Number of convolutional filters (also the viewpoint/aspect dimension ``f``).

    num_aspect: int, default: 5
        Number of viewpoints per user / aspects per item (``M`` in the paper).

    kernel_size: int, default: 3
        Convolution window size over the review document (``c`` in the paper).

    max_doc_length: int, default: 300
        Maximum number of tokens per user/item document.

    itr_self_attn: int, default: 2
        Number of iterations of the viewpoint self-attention routing.

    itr_routing: int, default: 3
        Number of Routing-by-Bi-Agreement iterations in the sentiment capsules
        (``tau`` in the paper).

    lambda_1: float, default: 0.8
        Weight of the mutual-exclusion (negative) term in the margin loss.

    gama: float, default: 0.5
        Multi-task trade-off (``lambda`` in the paper): ``gama*MSE + (1-gama)*margin``.

    rating_threshold: float, default: 3.0
        Ratings strictly above this are treated as positive sentiment.

    dropout_keep_prob: float, default: 0.9
        Keep probability of the DropConnect applied to weight matrices.

    batch_size: int, default: 100
        Batch size.

    max_iter: int, default: 150
        Max number of training epochs.

    learning_rate: float, default: 0.001
        Learning rate for the RMSProp optimizer.

    pretrained_w2v_path: str, optional, default: None
        Path to a pretrained word2vec/GloVe/fastText file used to initialize the
        word embeddings. If None, embeddings are randomly initialized (the official
        runner also random-inits) and a warning is emitted.

    pretrained_w2v_type: str, {'word2vec', 'glove', 'fasttext'}, default: 'word2vec'
        Format of `pretrained_w2v_path`.

    trainable: boolean, default: True
        When False, the model is not re-trained.

    verbose: boolean, default: True
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, e.g., a precomputed word-embedding matrix via
        init_params={'pretrained_word_embeddings': <numpy array>}. If provided,
        it takes precedence over `pretrained_w2v_path`.

    seed: int, optional, default: None
        Random seed for reproducibility.

    References
    ----------
    * Li, C., Quan, C., Peng, L., Qi, Y., Deng, Y., & Wu, L. (2019). A Capsule
      Network for Recommendation and Explaining What You Like and Dislike. SIGIR 2019.
    """

    def __init__(
        self,
        name="CARP",
        embedding_size=300,
        latent_dim=25,
        n_filters=50,
        num_aspect=5,
        kernel_size=3,
        max_doc_length=300,
        itr_self_attn=2,
        itr_routing=3,
        lambda_1=0.8,
        gama=0.5,
        rating_threshold=3.0,
        dropout_keep_prob=0.9,
        batch_size=100,
        max_iter=150,
        learning_rate=0.001,
        pretrained_w2v_path=None,
        pretrained_w2v_type="word2vec",
        trainable=True,
        verbose=True,
        init_params=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.embedding_size = embedding_size
        self.latent_dim = latent_dim
        self.n_filters = n_filters
        self.num_aspect = num_aspect
        self.kernel_size = kernel_size
        self.max_doc_length = max_doc_length
        self.itr_self_attn = itr_self_attn
        self.itr_routing = itr_routing
        self.lambda_1 = lambda_1
        self.gama = gama
        self.rating_threshold = rating_threshold
        self.dropout_keep_prob = dropout_keep_prob
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.pretrained_w2v_path = pretrained_w2v_path
        self.pretrained_w2v_type = pretrained_w2v_type
        self.init_params = {} if init_params is None else init_params
        self.seed = seed

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as the review modality.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes.

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        if not self.trainable:
            return self

        if train_set.review_text is None:
            raise ValueError("CARP requires a review modality (review_text).")

        self._fit_torch(train_set, val_set)
        return self

    def _fit_torch(self, train_set, val_set):
        import torch
        from .carp import CARPModel
        from .w2v_utils import build_w2v_matrix, build_doc_matrices

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vocab = train_set.review_text.vocab
        self.vocab_size = vocab.size
        # Rating bounds for the rescaled-sigmoid prediction head.
        self.min_rating = float(getattr(train_set, "min_rating", 1.0))
        self.max_rating = float(getattr(train_set, "max_rating", 5.0))

        # Per-user/item document token-id caches + word masks. Pad with the
        # dedicated zero padding row (index == vocab_size).
        self.user_doc, self.item_doc, self.user_mask, self.item_mask = build_doc_matrices(
            train_set.review_text,
            train_set.num_users,
            train_set.num_items,
            self.max_doc_length,
            pad_id=vocab.size,
        )

        # Word-embedding matrix (optional pretrained init). With no path, leave
        # pretrained=None so the model random-inits like the official runner.
        pretrained = self.init_params.get("pretrained_word_embeddings")
        if pretrained is None and self.pretrained_w2v_path is not None:
            pretrained, n_oov = build_w2v_matrix(
                vocab,
                path=self.pretrained_w2v_path,
                emb_type=self.pretrained_w2v_type,
                word_dim=self.embedding_size,
                seed=self.seed,
            )
            if self.verbose:
                print("[CARP] word embeddings: %d / %d OOV" % (n_oov, vocab.size))

        self.model = CARPModel(
            num_users=train_set.num_users,
            num_items=train_set.num_items,
            vocab_size=vocab.size,
            word_dim=self.embedding_size,
            latent_dim=self.latent_dim,
            n_filters=self.n_filters,
            num_aspect=self.num_aspect,
            kernel_size=self.kernel_size,
            itr_self_attn=self.itr_self_attn,
            itr_routing=self.itr_routing,
            min_rating=self.min_rating,
            max_rating=self.max_rating,
            dropout_keep_prob=self.dropout_keep_prob,
            pretrained_word_embeddings=pretrained,
        ).to(self.device)

        # Match TF RMSPropOptimizer defaults (decay=0.9, epsilon=1e-10); torch's
        # defaults (alpha=0.99, eps=1e-8) differ and would shift the trajectory.
        optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=self.learning_rate, alpha=0.9, eps=1e-10
        )
        mse = torch.nn.MSELoss()

        user_doc = torch.from_numpy(self.user_doc).to(self.device)
        item_doc = torch.from_numpy(self.item_doc).to(self.device)
        user_mask = torch.from_numpy(self.user_mask).to(self.device)
        item_mask = torch.from_numpy(self.item_mask).to(self.device)

        def _margin_loss(caps_len, labels):
            # labels: (bs,) 1.0 positive / 0.0 negative. Capsule 0 = pos, 1 = neg.
            T_c = torch.stack([labels, 1.0 - labels], dim=1)  # (bs, 2)
            max_p = torch.clamp(0.8 - caps_len, min=0.0) ** 2
            max_n = torch.clamp(caps_len - 0.2, min=0.0) ** 2
            L_c = T_c * max_p + self.lambda_1 * (1.0 - T_c) * max_n
            return torch.mean(torch.sum(L_c, dim=1))

        def _val_mse():
            self.model.eval()
            se, n = 0.0, 0
            with torch.no_grad():
                for bu, bi, br in val_set.uir_iter(self.batch_size, shuffle=False):
                    u = torch.from_numpy(bu).long().to(self.device)
                    i = torch.from_numpy(bi).long().to(self.device)
                    r = torch.from_numpy(br).float().to(self.device)
                    pred, _ = self.model(
                        u, i, user_doc[u], item_doc[i], user_mask[u], item_mask[i]
                    )
                    se += ((pred - r) ** 2).sum().item()
                    n += len(br)
            return se / max(n, 1)

        # Model selection: run the full epoch budget, keep the checkpoint with the
        # best validation MSE, restore it at the end (Cornac/iRev practice).
        best_val, best_state = float("inf"), None

        desc = "CARP lr=%g keep=%g M=%d bs=%d" % (
            self.learning_rate, self.dropout_keep_prob, self.num_aspect, self.batch_size,
        )
        n_batches = (train_set.num_ratings + self.batch_size - 1) // self.batch_size
        pbar = tqdm(total=self.max_iter * n_batches, disable=not self.verbose, desc=desc)
        for epoch in range(self.max_iter):
            self.model.train()
            sum_loss, count = 0.0, 0
            for batch_u, batch_i, batch_r in train_set.uir_iter(
                self.batch_size, shuffle=True
            ):
                u = torch.from_numpy(batch_u).long().to(self.device)
                i = torch.from_numpy(batch_i).long().to(self.device)
                r = torch.from_numpy(batch_r).float().to(self.device)
                labels = (r > self.rating_threshold).float()

                optimizer.zero_grad()
                pred, caps_len = self.model(
                    u, i, user_doc[u], item_doc[i], user_mask[u], item_mask[i]
                )
                rating_loss = mse(pred, r)
                margin_loss = _margin_loss(caps_len, labels)
                loss = self.gama * rating_loss + (1.0 - self.gama) * margin_loss
                loss.backward()
                optimizer.step()

                sum_loss += loss.item() * len(batch_r)
                count += len(batch_r)
                pbar.update(1)
                pbar.set_postfix(
                    ep=f"{epoch + 1}/{self.max_iter}",
                    loss=f"{sum_loss / max(count, 1):.4f}",
                )
            if val_set is not None:
                vmse = _val_mse()
                if vmse < best_val:
                    best_val, best_state = vmse, copy.deepcopy(self.model.state_dict())
                pbar.set_postfix(
                    ep=f"{epoch + 1}/{self.max_iter}",
                    loss=f"{sum_loss / max(count, 1):.4f}",
                    val=f"{vmse:.4f}",
                    best=f"{best_val:.4f}",
                )
        pbar.close()

        if best_state is not None:
            self.model.load_state_dict(best_state)  # restore best-validation epoch

        if self.verbose:
            print("Learning completed!")

    def _doc_tensors(self):
        """Device copies of the doc/mask matrices, built once and reused across
        score() calls (rating_eval calls score once per test user)."""
        import torch

        if getattr(self, "_doc_cache", None) is None:
            self._doc_cache = tuple(
                torch.from_numpy(arr).to(self.device)
                for arr in (self.user_doc, self.item_doc, self.user_mask, self.item_mask)
            )
        return self._doc_cache

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item (or all items).

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item. If None, scores for all known items are returned.

        Returns
        -------
        res : A scalar or a Numpy array
        """
        import torch

        if self.is_unknown_user(user_idx):
            raise ScoreException("Can't make score prediction for user %d" % user_idx)
        if item_idx is not None and self.is_unknown_item(item_idx):
            raise ScoreException("Can't make score prediction for item %d" % item_idx)

        self.model.eval()
        user_doc, item_doc, user_mask, item_mask = self._doc_tensors()
        with torch.no_grad():
            if item_idx is None:
                n_items = self.item_doc.shape[0]
                u = torch.full((n_items,), user_idx, dtype=torch.long, device=self.device)
                i = torch.arange(n_items, dtype=torch.long, device=self.device)
                pred, _ = self.model(
                    u, i, user_doc[u], item_doc[i], user_mask[u], item_mask[i]
                )
                return pred.cpu().numpy().ravel()
            else:
                u = torch.tensor([user_idx], dtype=torch.long, device=self.device)
                i = torch.tensor([item_idx], dtype=torch.long, device=self.device)
                pred, _ = self.model(
                    u, i, user_doc[u], item_doc[i], user_mask[u], item_mask[i]
                )
                return pred.item()

    def save(self, save_dir=None, save_trainset=False):
        """Save the model to the filesystem (state dict saved separately)."""
        import torch

        if save_dir is None:
            return

        self._doc_cache = None  # don't pickle device tensors
        model = self.model
        device = self.device
        del self.model
        del self.device

        model_file = Recommender.save(self, save_dir, save_trainset)

        self.model = model
        self.device = device
        torch.save(model.state_dict(), model_file.replace(".pkl", ".pt"))
        return model_file

    @staticmethod
    def load(model_path, trainable=False):
        """Load a model from the filesystem."""
        import torch
        from .carp import CARPModel

        model = Recommender.load(model_path, trainable)
        model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = CARPModel(
            num_users=model.num_users,
            num_items=model.num_items,
            vocab_size=model.vocab_size,
            word_dim=model.embedding_size,
            latent_dim=model.latent_dim,
            n_filters=model.n_filters,
            num_aspect=model.num_aspect,
            kernel_size=model.kernel_size,
            itr_self_attn=model.itr_self_attn,
            itr_routing=model.itr_routing,
            min_rating=model.min_rating,
            max_rating=model.max_rating,
            dropout_keep_prob=model.dropout_keep_prob,
        )
        net.load_state_dict(torch.load(model.load_from.replace(".pkl", ".pt")))
        model.model = net.to(model.device)
        return model
