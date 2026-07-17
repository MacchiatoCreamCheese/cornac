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


class MAN(Recommender):
    """Main-Auxiliary Network (MAN), faithful implementation from the paper.

    MAN separates the review streams per (user, item) pair: the **main network**
    processes RO (the user's reviews of other items) and ORT (other users' reviews
    of the target item) through Review Aspect Shift + Interaction-based Feature
    Learning + ID embeddings, producing predictive hidden features O. The
    **auxiliary network** processes RT (the target review) through word-based
    attention + multi-head self-attention + CNN/MLP + its own ID embeddings,
    producing accurate hidden features O*. Training follows Algorithm 1's three
    alternating steps: (J_A) fit the auxiliary net's rating head NFM2 on O*;
    (J_B) distill the main net's O toward O*; (J_C) fit the main rating head NFM1
    on O. Testing (Algorithm 2) uses the main network only.

    Parameters
    ----------
    name: string, default: 'MAN'
        The name of the recommender model.

    embedding_size: int, default: 300
        Word embedding size (paper uses 64-dim pretrained; 300 matches the
        benchmark's shared GoogleNews vectors). Must be divisible by ``n_heads``.

    id_embedding_size: int, default: 32
        User/item id embedding size (paper: latent feature dimension 32).

    n_filters: int, default: 100
        Number of convolutional filters (paper: 100).

    kernel_size: int, default: 3
        Convolution window size (paper: 3).

    fc_dim: int, default: 50
        Output dimension of the CNN text processor / interaction features
        (gamma in the paper: 50).

    att_hidden: int, default: 64
        Hidden size of the word-based attention network (k in Eq. 18).

    n_heads: int, default: 4
        Number of self-attention heads (paper: 4).

    ff_dim: int, default: 128
        Feed-forward dimension of the self-attention encoder layer (paper: 128).

    max_doc_length: int, default: 1000
        Maximum tokens of the RO / ORT streams (paper Sec. 5.4: max input text
        1,000 = 20 reviews x 50 words).

    max_rt_length: int, default: 50
        Maximum tokens of the RT stream (paper: single-review length 50).

    dropout_rate: float, default: 0.5
        Dropout rate (paper: keep probability 0.5).

    batch_size: int, default: 128
        Batch size (paper: 128).

    max_iter: int, default: 10
        Max number of training epochs (paper: best at 14-15).

    learning_rate: float, default: 0.006
        Learning rate for the three Adam optimizers (paper: 0.006).

    weight_decay: float, default: 0.001
        L2 regularization lambda (paper: 1e-3), applied via Adam weight decay.

    pretrained_w2v_path: str, optional, default: None
        Path to a pretrained word2vec/GloVe/fastText file for the word embeddings
        (held fixed during training). If None, embeddings are random and frozen.

    pretrained_w2v_type: str, {'word2vec', 'glove', 'fasttext'}, default: 'word2vec'
        Format of `pretrained_w2v_path`.

    trainable: boolean, default: True
        When False, the model is not re-trained.

    verbose: boolean, default: True
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        e.g. init_params={'pretrained_word_embeddings': <numpy array>}.

    seed: int, optional, default: None
        Random seed for reproducibility.

    References
    ----------
    * Yang, P., Xiao, Y., Zheng, W., Jiao, X., Zhu, K., Sun, C., & Liu, L. (2023).
      MAN: Main-auxiliary network with attentive interactions for review-based
      recommendation. Applied Intelligence 53:12955-12970.
    """

    def __init__(
        self,
        name="MAN",
        embedding_size=300,
        id_embedding_size=32,
        n_filters=100,
        kernel_size=3,
        fc_dim=50,
        att_hidden=64,
        n_heads=4,
        ff_dim=128,
        max_doc_length=1000,
        max_rt_length=50,
        dropout_rate=0.5,
        batch_size=128,
        max_iter=10,
        learning_rate=0.006,
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
        self.fc_dim = fc_dim
        self.att_hidden = att_hidden
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.max_doc_length = max_doc_length
        self.max_rt_length = max_rt_length
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
            raise ValueError("MAN requires a review modality (review_text).")

        self._fit_torch(train_set, val_set)
        return self

    def _batch_tensors(self, u, i, device):
        import torch

        ro, ort, rt = self.streams.build_batch(u, i)
        return (
            torch.from_numpy(ro).to(device),
            torch.from_numpy(ort).to(device),
            torch.from_numpy(rt).to(device),
        )

    def _fit_torch(self, train_set, val_set):
        import torch
        from .man import MANModel
        from .man_data import ReviewStreams
        from .w2v_utils import build_w2v_matrix

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vocab = train_set.review_text.vocab
        self.vocab_size = vocab.size
        self.streams = ReviewStreams(
            train_set.review_text, self.max_doc_length, self.max_rt_length
        )

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
                print("[MAN] word embeddings: %d / %d OOV" % (n_oov, vocab.size))

        self.model = MANModel(
            num_users=train_set.num_users,
            num_items=train_set.num_items,
            vocab_size=vocab.size,
            word_dim=self.embedding_size,
            id_embedding_size=self.id_embedding_size,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            fc_dim=self.fc_dim,
            att_hidden=self.att_hidden,
            n_heads=self.n_heads,
            ff_dim=self.ff_dim,
            dropout_rate=self.dropout_rate,
            pretrained_word_embeddings=pretrained,
        ).to(self.device)

        # Three optimizers over disjoint parameter groups (Algorithm 1):
        # theta_A = auxiliary net + NFM2; theta_B = main feature extractor;
        # theta_C = NFM1. L2 regularization lambda via Adam weight decay.
        # The shared word-embedding table is frozen (paper Eq. 1) and excluded.
        def _trainable(*modules):
            return [p for md in modules for p in md.parameters() if p.requires_grad]

        opt_aux = torch.optim.Adam(
            _trainable(self.model.aux_net, self.model.nfm2),
            lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        opt_main = torch.optim.Adam(
            _trainable(self.model.main_net),
            lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        opt_nfm1 = torch.optim.Adam(
            _trainable(self.model.nfm1),
            lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        mse = torch.nn.MSELoss()

        def _val_mse():
            self.model.eval()
            se, n = 0.0, 0
            with torch.no_grad():
                for bu, bi, br in val_set.uir_iter(self.batch_size, shuffle=False):
                    u = torch.from_numpy(bu).long().to(self.device)
                    i = torch.from_numpy(bi).long().to(self.device)
                    r = torch.from_numpy(br).float().to(self.device)
                    ro, ort, _ = self._batch_tensors(bu, bi, self.device)
                    pred = self.model(u, i, ro, ort)
                    se += ((pred - r) ** 2).sum().item()
                    n += len(br)
            return se / max(n, 1)

        best_val, best_state = float("inf"), None

        desc = "MAN lr=%g wd=%g bs=%d" % (
            self.learning_rate, self.weight_decay, self.batch_size,
        )
        n_batches = (train_set.num_ratings + self.batch_size - 1) // self.batch_size
        pbar = tqdm(total=self.max_iter * n_batches, disable=not self.verbose, desc=desc)
        for epoch in range(self.max_iter):
            self.model.train()
            sum_a, sum_b, sum_c, count = 0.0, 0.0, 0.0, 0
            for batch_u, batch_i, batch_r in train_set.uir_iter(
                self.batch_size, shuffle=True
            ):
                u = torch.from_numpy(batch_u).long().to(self.device)
                i = torch.from_numpy(batch_i).long().to(self.device)
                r = torch.from_numpy(batch_r).float().to(self.device)
                ro, ort, rt = self._batch_tensors(batch_u, batch_i, self.device)

                # ---- Step 1 (J_A): train the auxiliary network on RT. ----
                opt_aux.zero_grad()
                o_star = self.model.aux_net(u, i, rt)
                loss_a = mse(self.model.nfm2(o_star).squeeze(1), r)
                loss_a.backward()
                opt_aux.step()

                # ---- Step 2 (J_B): distill O toward the (updated) O*. ----
                with torch.no_grad():
                    o_star_label = self.model.aux_net(u, i, rt)
                opt_main.zero_grad()
                o = self.model.main_net(u, i, ro, ort)
                loss_b = mse(o, o_star_label)
                loss_b.backward()
                opt_main.step()

                # ---- Step 3 (J_C): train NFM1 on the (detached) O. ----
                opt_nfm1.zero_grad()
                pred = self.model.nfm1(o.detach()).squeeze(1)
                loss_c = mse(pred, r)
                loss_c.backward()
                opt_nfm1.step()

                bs = len(batch_r)
                sum_a += loss_a.item() * bs
                sum_b += loss_b.item() * bs
                sum_c += loss_c.item() * bs
                count += bs
                pbar.update(1)
                pbar.set_postfix(
                    ep=f"{epoch + 1}/{self.max_iter}",
                    JA=f"{sum_a / max(count, 1):.3f}",
                    JB=f"{sum_b / max(count, 1):.3f}",
                    JC=f"{sum_c / max(count, 1):.3f}",
                )

            if val_set is not None:
                vmse = _val_mse()
                if vmse < best_val:
                    best_val, best_state = vmse, copy.deepcopy(self.model.state_dict())
                pbar.set_postfix(
                    ep=f"{epoch + 1}/{self.max_iter}",
                    JC=f"{sum_c / max(count, 1):.3f}",
                    val=f"{vmse:.4f}",
                    best=f"{best_val:.4f}",
                )
        pbar.close()

        if best_state is not None:
            self.model.load_state_dict(best_state)  # restore best-validation epoch

        if self.verbose:
            print("Learning completed!")

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item (or all items).

        Uses the main network only (Algorithm 2)."""
        import torch

        if self.is_unknown_user(user_idx):
            raise ScoreException("Can't make score prediction for user %d" % user_idx)
        if item_idx is not None and self.is_unknown_item(item_idx):
            raise ScoreException("Can't make score prediction for item %d" % item_idx)

        self.model.eval()
        with torch.no_grad():
            if item_idx is None:
                n_items = self.num_items
                preds = []
                for start in range(0, n_items, self.batch_size):
                    end = min(start + self.batch_size, n_items)
                    bu = np.full(end - start, user_idx, dtype=np.int64)
                    bi = np.arange(start, end, dtype=np.int64)
                    u = torch.from_numpy(bu).to(self.device)
                    i = torch.from_numpy(bi).to(self.device)
                    ro, ort, _ = self._batch_tensors(bu, bi, self.device)
                    preds.append(self.model(u, i, ro, ort).cpu().numpy())
                return np.concatenate(preds).ravel()
            else:
                bu = np.array([user_idx], dtype=np.int64)
                bi = np.array([item_idx], dtype=np.int64)
                u = torch.from_numpy(bu).to(self.device)
                i = torch.from_numpy(bi).to(self.device)
                ro, ort, _ = self._batch_tensors(bu, bi, self.device)
                return self.model(u, i, ro, ort).item()

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
        from .man import MANModel

        model = Recommender.load(model_path, trainable)
        model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = MANModel(
            num_users=model.num_users,
            num_items=model.num_items,
            vocab_size=model.vocab_size,
            word_dim=model.embedding_size,
            id_embedding_size=model.id_embedding_size,
            n_filters=model.n_filters,
            kernel_size=model.kernel_size,
            fc_dim=model.fc_dim,
            att_hidden=model.att_hidden,
            n_heads=model.n_heads,
            ff_dim=model.ff_dim,
            dropout_rate=model.dropout_rate,
        )
        net.load_state_dict(torch.load(model.load_from.replace(".pkl", ".pt")))
        model.model = net.to(model.device)
        return model
