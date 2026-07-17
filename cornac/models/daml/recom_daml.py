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


class DAML(Recommender):
    """Dual Attention Mutual Learning (DAML).

    A review-based rating-prediction model that learns local word-level attention
    with a shared CNN, couples user and item document features through a mutual
    (Euclidean) attention matrix, and pools them with an unfold-based local pooling.
    The review and id features are fused by addition per side (paper Eq. 15),
    concatenated (Eq. 16), and scored with a Neural Factorization Machine head
    (Eq. 17-19) -- the paper-faithful configuration (following the Neu-Review-Rec
    reference), replacing the iRev benchmark's default concat + LFM head.

    Parameters
    ----------
    name: string, default: 'DAML'
        The name of the recommender model.

    embedding_size: int, default: 300
        Word embedding size.

    id_embedding_size: int, default: 32
        User/item id embedding size.

    n_filters: int, default: 100
        Number of convolutional filters.

    kernel_size: int, default: 3
        Convolution window size.

    max_doc_length: int, default: 500
        Maximum number of tokens per user/item document.

    dropout_rate: float, default: 0.5
        Dropout rate before the prediction head.

    batch_size: int, default: 8
        Batch size (DAML is memory-heavy due to the mutual attention matrix).

    max_iter: int, default: 10
        Max number of training epochs.

    learning_rate: float, default: 0.002
        Learning rate for the Adam optimizer.

    weight_decay: float, default: 0.001
        L2 weight decay for the Adam optimizer.

    pretrained_w2v_path: str, optional, default: None
        Path to a pretrained word2vec/GloVe/fastText file used to initialize the
        word embeddings (iRev-faithful). If None, embeddings are randomly
        initialized and a warning is emitted.

    pretrained_w2v_type: str, {'word2vec', 'glove', 'fasttext'}, default: 'word2vec'
        Format of `pretrained_w2v_path`.

    trainable: boolean, default: True
        When False, the model is not re-trained.

    verbose: boolean, default: True
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, e.g., init_params={'pretrained_word_embeddings': <numpy array>}.
        If provided, it takes precedence over `pretrained_w2v_path`.

    seed: int, optional, default: None
        Random seed for reproducibility.

    References
    ----------
    * Liu, D., Li, J., Du, B., Chang, J., & Gao, R. (2019). DAML: Dual Attention
      Mutual Learning between Ratings and Reviews for Item Recommendation. KDD 2019.
    """

    def __init__(
        self,
        name="DAML",
        embedding_size=300,
        id_embedding_size=32,
        n_filters=100,
        kernel_size=3,
        max_doc_length=500,
        dropout_rate=0.5,
        batch_size=8,
        max_iter=10,
        learning_rate=0.002,
        weight_decay=0.001,
        pretrained_w2v_path=None,
        pretrained_w2v_type="word2vec",
        trainable=True,
        verbose=True,
        init_params=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.embedding_size = embedding_size
        self.id_embedding_size = id_embedding_size
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.max_doc_length = max_doc_length
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
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
            raise ValueError("DAML requires a review modality (review_text).")

        self._fit_torch(train_set, val_set)
        return self

    def _fit_torch(self, train_set, val_set):
        import torch
        from .daml import DAMLModel
        from .w2v_utils import build_w2v_matrix, build_doc_matrices

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vocab = train_set.review_text.vocab
        self.vocab_size = vocab.size
        self.user_doc, self.item_doc = build_doc_matrices(
            train_set.review_text,
            train_set.num_users,
            train_set.num_items,
            self.max_doc_length,
        )

        pretrained = self.init_params.get("pretrained_word_embeddings")
        if pretrained is None:
            pretrained, n_oov = build_w2v_matrix(
                vocab,
                path=self.pretrained_w2v_path,
                emb_type=self.pretrained_w2v_type,
                word_dim=self.embedding_size,
                seed=self.seed,
            )
            if self.verbose and self.pretrained_w2v_path is not None:
                print("[DAML] word embeddings: %d / %d OOV" % (n_oov, vocab.size))

        self.model = DAMLModel(
            num_users=train_set.num_users,
            num_items=train_set.num_items,
            vocab_size=vocab.size,
            word_dim=self.embedding_size,
            id_embedding_size=self.id_embedding_size,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            dropout_rate=self.dropout_rate,
            pretrained_word_embeddings=pretrained,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
        criterion = torch.nn.MSELoss()

        user_doc = torch.from_numpy(self.user_doc).to(self.device)
        item_doc = torch.from_numpy(self.item_doc).to(self.device)

        def _val_mse():
            self.model.eval()
            se, n = 0.0, 0
            with torch.no_grad():
                for bu, bi, br in val_set.uir_iter(self.batch_size, shuffle=False):
                    u = torch.from_numpy(bu).long().to(self.device)
                    i = torch.from_numpy(bi).long().to(self.device)
                    r = torch.from_numpy(br).float().to(self.device)
                    pred = self.model(u, i, user_doc[u], item_doc[i])
                    se += ((pred - r) ** 2).sum().item()
                    n += len(br)
            return se / max(n, 1)

        # iRev-style model selection: run the full epoch budget, keep the
        # checkpoint with the best validation MSE, restore it at the end.
        best_val, best_state = float("inf"), None

        desc = "DAML lr=%g do=%g id=%d wd=%g bs=%d" % (
            self.learning_rate, self.dropout_rate, self.id_embedding_size,
            self.weight_decay, self.batch_size,
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

                optimizer.zero_grad()
                pred = self.model(u, i, user_doc[u], item_doc[i])
                loss = criterion(pred, r)
                loss.backward()
                optimizer.step()

                sum_loss += loss.item() * len(batch_r)
                count += len(batch_r)
                pbar.update(1)
                pbar.set_postfix(
                    ep=f"{epoch + 1}/{self.max_iter}",
                    loss=f"{sum_loss / max(count, 1):.4f}",
                )
            scheduler.step()

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

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item (or all items)."""
        import torch

        if self.is_unknown_user(user_idx):
            raise ScoreException("Can't make score prediction for user %d" % user_idx)
        if item_idx is not None and self.is_unknown_item(item_idx):
            raise ScoreException("Can't make score prediction for item %d" % item_idx)

        self.model.eval()
        user_doc = torch.from_numpy(self.user_doc).to(self.device)
        item_doc = torch.from_numpy(self.item_doc).to(self.device)
        with torch.no_grad():
            if item_idx is None:
                n_items = self.item_doc.shape[0]
                u = torch.full((n_items,), user_idx, dtype=torch.long, device=self.device)
                i = torch.arange(n_items, dtype=torch.long, device=self.device)
                # Batch to bound memory (mutual attention matrix is O(L^2)).
                preds = []
                for start in range(0, n_items, self.batch_size):
                    end = min(start + self.batch_size, n_items)
                    bu, bi = u[start:end], i[start:end]
                    preds.append(
                        self.model(bu, bi, user_doc[bu], item_doc[bi]).cpu().numpy()
                    )
                return np.concatenate(preds).ravel()
            else:
                u = torch.tensor([user_idx], dtype=torch.long, device=self.device)
                i = torch.tensor([item_idx], dtype=torch.long, device=self.device)
                pred = self.model(u, i, user_doc[u], item_doc[i])
                return pred.item()

    def save(self, save_dir=None, save_trainset=False):
        """Save the model to the filesystem (state dict saved separately)."""
        import torch

        if save_dir is None:
            return

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
        from .daml import DAMLModel

        model = Recommender.load(model_path, trainable)
        model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = DAMLModel(
            num_users=model.num_users,
            num_items=model.num_items,
            vocab_size=model.vocab_size,
            word_dim=model.embedding_size,
            id_embedding_size=model.id_embedding_size,
            n_filters=model.n_filters,
            kernel_size=model.kernel_size,
            dropout_rate=model.dropout_rate,
        )
        net.load_state_dict(torch.load(model.load_from.replace(".pkl", ".pt")))
        model.model = net.to(model.device)
        return model
