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

import numpy as np
from tqdm.auto import trange

from ..recommender import Recommender
from ...exception import ScoreException


class ALFM(Recommender):
    """Aspect-Aware Latent Factor Model (ALFM).

    A review-based rating-prediction model that represents each user/item review
    document by an LDA aspect (topic) distribution, fuses it with id embeddings
    through an attentive interaction, and predicts the rating directly. Unlike the
    other ported models, ALFM uses global (topic) features rather than word
    embeddings. Ported to Cornac (PyTorch) from the iRev benchmark implementation.

    Parameters
    ----------
    name: string, default: 'ALFM'
        The name of the recommender model.

    id_embedding_size: int, default: 32
        User/item id embedding size.

    n_topics: int, default: 32
        Number of LDA aspects/topics used to represent each document.

    max_doc_length: int, default: 500
        Maximum number of tokens per user/item document (for the topic model).

    dropout_rate: float, default: 0.5
        Dropout rate in the rating-prediction MLP.

    batch_size: int, default: 128
        Batch size.

    max_iter: int, default: 10
        Max number of training epochs.

    learning_rate: float, default: 0.002
        Learning rate for the Adam optimizer.

    weight_decay: float, default: 0.001
        L2 weight decay for the Adam optimizer.

    lda_max_iter: int, default: 10
        Number of iterations for the LDA topic model.

    trainable: boolean, default: True
        When False, the model is not re-trained.

    verbose: boolean, default: True
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        Unused placeholder kept for API consistency with the other review models.

    seed: int, optional, default: None
        Random seed for reproducibility.

    References
    ----------
    * Cheng, Z., Ding, Y., Zhu, L., & Kankanhalli, M. (2018). Aspect-Aware Latent
      Factor Model: Rating Prediction with Ratings and Reviews. WWW 2018.
    """

    def __init__(
        self,
        name="ALFM",
        id_embedding_size=32,
        n_topics=32,
        max_doc_length=500,
        dropout_rate=0.5,
        batch_size=128,
        max_iter=10,
        learning_rate=0.002,
        weight_decay=0.001,
        lda_max_iter=10,
        trainable=True,
        verbose=True,
        init_params=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.id_embedding_size = id_embedding_size
        self.n_topics = n_topics
        self.max_doc_length = max_doc_length
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lda_max_iter = lda_max_iter
        self.init_params = {} if init_params is None else init_params
        self.seed = seed

    def _build_topics(self, train_set):
        """Fit an LDA model and return per-user / per-item topic matrices."""
        from sklearn.decomposition import LatentDirichletAllocation
        from .w2v_utils import build_doc_matrices

        vocab = train_set.review_text.vocab
        user_doc, item_doc = build_doc_matrices(
            train_set.review_text,
            train_set.num_users,
            train_set.num_items,
            self.max_doc_length,
        )

        def _bow(doc_matrix):
            counts = np.zeros((doc_matrix.shape[0], vocab.size), dtype="float32")
            for row in range(doc_matrix.shape[0]):
                np.add.at(counts[row], doc_matrix[row], 1.0)
            counts[:, :4] = 0.0  # drop special tokens (<PAD>/<UNK>/<BOS>/<EOS>)
            return counts

        user_bow = _bow(user_doc)
        item_bow = _bow(item_doc)

        lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=self.lda_max_iter,
            learning_method="online",
            random_state=self.seed,
        )
        lda.fit(np.vstack([user_bow, item_bow]))
        user_topics = lda.transform(user_bow).astype("float32")
        item_topics = lda.transform(item_bow).astype("float32")
        return user_topics, item_topics

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
            raise ValueError("ALFM requires a review modality (review_text).")

        self._fit_torch(train_set, val_set)
        return self

    def _fit_torch(self, train_set, val_set):
        import torch
        from .alfm import ALFMModel

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.verbose:
            print("[ALFM] fitting LDA topic model (%d topics)..." % self.n_topics)
        self.user_topics, self.item_topics = self._build_topics(train_set)

        self.model = ALFMModel(
            num_users=train_set.num_users,
            num_items=train_set.num_items,
            topic_dim=self.n_topics,
            id_embedding_size=self.id_embedding_size,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
        criterion = torch.nn.MSELoss()

        user_topics = torch.from_numpy(self.user_topics).to(self.device)
        item_topics = torch.from_numpy(self.item_topics).to(self.device)

        loop = trange(self.max_iter, disable=not self.verbose)
        for _ in loop:
            self.model.train()
            sum_loss, count = 0.0, 0
            for batch_u, batch_i, batch_r in train_set.uir_iter(
                self.batch_size, shuffle=True
            ):
                u = torch.from_numpy(batch_u).long().to(self.device)
                i = torch.from_numpy(batch_i).long().to(self.device)
                r = torch.from_numpy(batch_r).float().to(self.device)

                optimizer.zero_grad()
                pred = self.model(u, i, user_topics[u], item_topics[i])
                loss = criterion(pred, r)
                loss.backward()
                optimizer.step()

                sum_loss += loss.item() * len(batch_r)
                count += len(batch_r)
            scheduler.step()
            if self.verbose:
                loop.set_postfix(loss=sum_loss / max(count, 1))
        loop.close()

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
        user_topics = torch.from_numpy(self.user_topics).to(self.device)
        item_topics = torch.from_numpy(self.item_topics).to(self.device)
        with torch.no_grad():
            if item_idx is None:
                n_items = self.item_topics.shape[0]
                u = torch.full((n_items,), user_idx, dtype=torch.long, device=self.device)
                i = torch.arange(n_items, dtype=torch.long, device=self.device)
                preds = self.model(u, i, user_topics[u], item_topics[i])
                return preds.cpu().numpy().ravel()
            else:
                u = torch.tensor([user_idx], dtype=torch.long, device=self.device)
                i = torch.tensor([item_idx], dtype=torch.long, device=self.device)
                pred = self.model(u, i, user_topics[u], item_topics[i])
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
        from .alfm import ALFMModel

        model = Recommender.load(model_path, trainable)
        model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = ALFMModel(
            num_users=model.num_users,
            num_items=model.num_items,
            topic_dim=model.n_topics,
            id_embedding_size=model.id_embedding_size,
            dropout_rate=model.dropout_rate,
        )
        net.load_state_dict(torch.load(model.load_from.replace(".pkl", ".pt")))
        model.model = net.to(model.device)
        return model
