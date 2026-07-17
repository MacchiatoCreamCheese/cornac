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
"""NumPy prediction math for the faithful ALFM (aspect-aware latent factor model).

Rating (MFRecommender.predict):
    r(u,i) = sum_a topicPart(u,i,a) * sum_f w[a,f]^2 * U[u,f] * V[i,f]
             + b_u + b_i + mean
with topicPart(u,i,a) = (pi_u*lambdaU[u,a] + (1-pi_u)*lambdaV[i,a]) * (1 - JSD(thetaU[u,a], thetaV[i,a])).

Because the topic model smooths every distribution with the Dirichlet prior alpha,
all thetaU/thetaV entries are strictly positive, so the exact ``KLDis`` edge cases in
getTopicPartFactor.java (u_k==0 skip / v_k==0 -> big) never fire and the plain
base-2 Jensen-Shannon divergence used here is numerically identical.
"""

import numpy as np


def _jsd_sim(ta, tb):
    """1 - JSD(ta, tb) with base-2 log, broadcast over the leading axes (last axis = K)."""
    m = 0.5 * (ta + tb)
    kl1 = np.sum(ta * np.log2(ta / m), axis=-1)
    kl2 = np.sum(tb * np.log2(tb / m), axis=-1)
    return 1.0 - 0.5 * (kl1 + kl2)


def topic_part_batch(u_idx, i_idx, pi, lambda_u, lambda_v, theta_u, theta_v):
    """Per-rating topicPart, shape (N, A). u_idx/i_idx are int arrays of length N."""
    tu = theta_u[u_idx]  # (N, A, K)
    tv = theta_v[i_idx]  # (N, A, K)
    s = _jsd_sim(tu, tv)  # (N, A)
    p = pi[u_idx][:, None]  # (N, 1)
    ratio = p * lambda_u[u_idx] + (1.0 - p) * lambda_v[i_idx]  # (N, A)
    return ratio * s


def topic_part_user_items(u, pi, lambda_u, lambda_v, theta_u, theta_v):
    """topicPart of user ``u`` against ALL items, shape (num_items, A)."""
    tu = theta_u[u][None, :, :]  # (1, A, K)
    s = _jsd_sim(tu, theta_v)  # (num_items, A)
    ratio = pi[u] * lambda_u[u][None, :] + (1.0 - pi[u]) * lambda_v  # (num_items, A)
    return ratio * s


def predict_user_items(u, params):
    """Predict ratings of user ``u`` for all items, shape (num_items,)."""
    tp = topic_part_user_items(
        u, params["pi"], params["lambda_u"], params["lambda_v"],
        params["theta_u"], params["theta_v"],
    )  # (I, A)
    w2 = params["w"] ** 2  # (A, F)
    uf = params["user_factors"][u]  # (F,)
    coeff = w2 * uf[None, :]  # (A, F)
    # einsum (not '@'): avoids a BLAS matmul that can crash under some Windows
    # numpy/threading setups; aspect_rate[i,a] = sum_f item_factors[i,f]*coeff[a,f].
    aspect_rate = np.einsum("if,af->ia", params["item_factors"], coeff)  # (I, A)
    pred = np.sum(tp * aspect_rate, axis=1)
    pred += params["user_bias"][u] + params["item_bias"] + params["mean"]
    return pred


def predict_single(u, i, params):
    """Predict the rating of user ``u`` for item ``i`` (scalar)."""
    tu = params["theta_u"][u]  # (A, K)
    tv = params["theta_v"][i]  # (A, K)
    s = _jsd_sim(tu, tv)  # (A,)
    ratio = params["pi"][u] * params["lambda_u"][u] + (1.0 - params["pi"][u]) * params["lambda_v"][i]
    tp = ratio * s  # (A,)
    w2 = params["w"] ** 2  # (A, F)
    aspect_rate = np.sum(w2 * params["user_factors"][u][None, :] * params["item_factors"][i][None, :], axis=1)
    pred = float(np.sum(tp * aspect_rate))
    pred += params["user_bias"][u] + params["item_bias"][i] + params["mean"]
    return pred
