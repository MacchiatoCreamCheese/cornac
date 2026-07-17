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

from ..recommender import Recommender
from ...exception import ScoreException


class ALFM(Recommender):
    """Aspect-Aware Latent Factor Model (ALFM).

    Faithful Cornac port of the authors' official Java implementation
    (WWW 2018). ALFM works in two stages: (1) an aspect-aware topic model (ATM) is
    fit by collapsed Gibbs sampling over sentence-segmented reviews, producing, for
    every user/item and aspect, a topic distribution (thetaU/thetaV), an aspect
    importance (lambdaU/lambdaV) and a user Bernoulli prior (pi); (2) an aspect-aware
    latent-factor model is fit by SGD, where the overall rating is a linear
    combination of aspect ratings weighted by aspect importance x aspect similarity
    (1 - JSD of the topic distributions).

    Parameters
    ----------
    name: string, default: 'ALFM'
        The name of the recommender model.

    num_aspects: int, default: 5
        Number of aspects (``A`` in the paper).

    n_topics: int, default: 5
        Number of latent topics per aspect (``K`` in the paper).

    num_factors: int, default: 5
        Number of latent factors (``f`` in the paper).

    alpha, beta, gamma: float, default: 0.1, 0.01, 0.5
        Dirichlet priors of the topic model (topic, word, aspect respectively).

    eta: tuple(float, float), default: (1.0, 1.0)
        Beta prior for the user/item Bernoulli source variable.

    tm_iterations: int, default: 120
        Number of Gibbs iterations for the topic model.

    tm_begin_save, tm_save_step: int, default: 100, 10
        Iteration at which parameter estimation starts, and the estimation period.
        The estimate from the last estimation iteration is kept.

    learn_rate: float, default: 0.01
        SGD learning rate (constant; no bold-driver, no decay).

    reg: float, default: 0.5
        L2 regularization for user/item factors and biases.

    weight_reg: float, default: 0.01
        L1 (smoothed) regularization for the aspect weight matrix.

    epsilon: float, default: 1e-6
        Smoothing constant for the L1 gradient ``|x| ~ sqrt(x^2 + epsilon)``.

    max_iter: int, default: 20
        Maximum number of SGD iterations (early-stops on loss convergence).

    trainable: boolean, default: True
        When False, the model is not re-trained.

    verbose: boolean, default: True
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        Used to inject/cache the ATM outputs so the tuner can reuse them across
        factor-only sweeps (keyed by ``atm_{A}_{K}``) and to cache the parsed docs
        (``docs``). Also accepts a precomputed model.

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
        num_aspects=5,
        n_topics=5,
        num_factors=5,
        alpha=0.1,
        beta=0.01,
        gamma=0.5,
        eta=(1.0, 1.0),
        tm_iterations=120,
        tm_begin_save=100,
        tm_save_step=10,
        learn_rate=0.01,
        reg=0.5,
        weight_reg=0.01,
        epsilon=1e-6,
        max_iter=20,
        trainable=True,
        verbose=True,
        init_params=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.num_aspects = num_aspects
        self.n_topics = n_topics
        self.num_factors = num_factors
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.tm_iterations = tm_iterations
        self.tm_begin_save = tm_begin_save
        self.tm_save_step = tm_save_step
        self.learn_rate = learn_rate
        self.reg = reg
        self.weight_reg = weight_reg
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.init_params = {} if init_params is None else init_params
        self.seed = seed

    # ------------------------------------------------------------------ ATM
    def _get_docs(self, train_set):
        """Build (and cache in init_params) the sentence-segmented documents."""
        docs = self.init_params.get("docs")
        if docs is None:
            from .alfm_data import build_documents

            if self.verbose:
                print("[ALFM] segmenting reviews into sentences...")
            docs = build_documents(
                train_set.review_text, train_set.num_users, train_set.num_items
            )
            self.init_params["docs"] = docs
        return docs

    def build_atm(self, train_set):
        """Fit the ATM topic model; return dict of thetaU/thetaV/lambdaU/lambdaV/pi.

        Exposed so a tuner can precompute it once per (num_aspects, n_topics) and
        inject it via ``init_params`` (see ``_atm_key``).
        """
        from ._alfm import run_atm

        docs = self._get_docs(train_set)
        maxW = 0
        swp = docs["sent_word_ptr"]
        if len(swp) > 1:
            maxW = int(np.max(swp[1:] - swp[:-1]))
        maxW = max(maxW, 1)

        if self.verbose:
            print(
                "[ALFM] fitting ATM (A=%d, K=%d, %d docs, %d words, vocab=%d)..."
                % (self.num_aspects, self.n_topics, len(docs["doc_user"]),
                   len(docs["words"]), docs["vocab_size"])
            )
        theta_u, theta_v, lambda_u, lambda_v, pi = run_atm(
            docs["num_users"], docs["num_items"], self.num_aspects, self.n_topics,
            docs["vocab_size"], docs["doc_user"], docs["doc_item"],
            docs["doc_sent_ptr"], docs["sent_word_ptr"], docs["words"],
            float(self.alpha), float(self.beta), float(self.gamma),
            float(self.eta[0]), float(self.eta[1]),
            int(self.tm_iterations), int(self.tm_begin_save), int(self.tm_save_step),
            int(maxW), np.uint64(self.seed if self.seed is not None else 88172645463325252),
            1 if self.verbose else 0,
        )
        return {
            "theta_u": theta_u, "theta_v": theta_v,
            "lambda_u": lambda_u, "lambda_v": lambda_v, "pi": pi,
        }

    def _atm_key(self):
        return "atm_%d_%d" % (self.num_aspects, self.n_topics)

    def _get_atm(self, train_set):
        key = self._atm_key()
        atm = self.init_params.get(key)
        if atm is None:
            atm = self.build_atm(train_set)
            self.init_params[key] = atm
        return atm

    # ------------------------------------------------------------------ fit
    def fit(self, train_set, val_set=None):
        Recommender.fit(self, train_set, val_set)
        if not self.trainable:
            return self
        if train_set.review_text is None:
            raise ValueError("ALFM requires a review modality (review_text).")

        rng = np.random.RandomState(self.seed)
        atm = self._get_atm(train_set)
        self.theta_u = atm["theta_u"]
        self.theta_v = atm["theta_v"]
        self.lambda_u = atm["lambda_u"]
        self.lambda_v = atm["lambda_v"]
        self.pi = atm["pi"]

        # training ratings
        uir_u, uir_i, uir_r = train_set.uir_tuple
        rating_user = np.ascontiguousarray(uir_u, dtype=np.int32)
        rating_item = np.ascontiguousarray(uir_i, dtype=np.int32)
        ratings = np.ascontiguousarray(uir_r, dtype=np.float64)
        self.mean = float(ratings.mean())

        # per-rating topicPart (N, A)
        from .alfm import topic_part_batch

        topic_part = np.ascontiguousarray(
            topic_part_batch(rating_user, rating_item, self.pi, self.lambda_u,
                             self.lambda_v, self.theta_u, self.theta_v),
            dtype=np.float64,
        )

        # init factors / weights / biases  ~ N(0, 0.1)
        A, F = self.num_aspects, self.num_factors
        n_users, n_items = train_set.num_users, train_set.num_items
        self.user_factors = np.ascontiguousarray(rng.normal(0, 0.1, (n_users, F)), dtype=np.float64)
        self.item_factors = np.ascontiguousarray(rng.normal(0, 0.1, (n_items, F)), dtype=np.float64)
        self.w = np.ascontiguousarray(rng.normal(0, 0.1, (A, F)), dtype=np.float64)
        self.user_bias = np.ascontiguousarray(rng.normal(0, 0.1, n_users), dtype=np.float64)
        self.item_bias = np.ascontiguousarray(rng.normal(0, 0.1, n_items), dtype=np.float64)

        from ._alfm import run_sgd

        if self.verbose:
            print("[ALFM] fitting latent factors (SGD, f=%d)..." % F)
        run_sgd(
            n_users, n_items, A, F,
            rating_user, rating_item, ratings, topic_part,
            self.user_factors, self.item_factors, self.w,
            self.user_bias, self.item_bias,
            float(self.learn_rate), float(self.reg), float(self.weight_reg),
            float(self.epsilon), self.mean, int(self.max_iter), 1 if self.verbose else 0,
        )
        if self.verbose:
            print("Learning completed!")
        return self

    def _params(self):
        return {
            "theta_u": self.theta_u, "theta_v": self.theta_v,
            "lambda_u": self.lambda_u, "lambda_v": self.lambda_v, "pi": self.pi,
            "w": self.w, "user_factors": self.user_factors,
            "item_factors": self.item_factors, "user_bias": self.user_bias,
            "item_bias": self.item_bias, "mean": self.mean,
        }

    # ---------------------------------------------------------------- score
    def score(self, user_idx, item_idx=None):
        from .alfm import predict_user_items, predict_single

        if self.is_unknown_user(user_idx):
            raise ScoreException("Can't make score prediction for user %d" % user_idx)
        if item_idx is not None and self.is_unknown_item(item_idx):
            raise ScoreException("Can't make score prediction for item %d" % item_idx)

        params = self._params()
        if item_idx is None:
            return predict_user_items(user_idx, params)
        return predict_single(user_idx, item_idx, params)

    # ----------------------------------------------------------- save/load
    def save(self, save_dir=None, save_trainset=False):
        """Save the model. The bulky doc/ATM caches in init_params are stripped
        (the trained arrays needed for scoring are kept as attributes)."""
        if save_dir is None:
            return
        cached = self.init_params
        self.init_params = {}
        try:
            model_file = Recommender.save(self, save_dir, save_trainset)
        finally:
            self.init_params = cached
        return model_file
