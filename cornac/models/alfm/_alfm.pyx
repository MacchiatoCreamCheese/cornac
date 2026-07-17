# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
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
"""Cython core for the faithful ALFM port.

    run_atm : sentence-level aspect-aware topic model (collapsed Gibbs sampling),
              faithful to ALFM/src/topicmodel/aspectTopicModel.java.
    run_sgd : aspect-aware latent-factor model (per-rating SGD), faithful to
              ALFM/src/alfm/MFRecommender.java.
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport log, exp, fabs, sqrt, pow


cdef inline double _rand_double(unsigned long long* state) nogil:
    """xorshift64* uniform in [0, 1) -- avoids MSVC's tiny RAND_MAX."""
    cdef unsigned long long x = state[0]
    x ^= x >> 12
    x ^= x << 25
    x ^= x >> 27
    state[0] = x
    return ((x * <unsigned long long>2685821657736338717) >> 11) * (1.0 / 9007199254740992.0)


def run_atm(int num_users, int num_items, int A, int K, int V,
            int[::1] doc_user, int[::1] doc_item,
            int[::1] doc_sent_ptr, int[::1] sent_word_ptr, int[::1] words,
            double alpha, double beta, double gamma, double eta0, double eta1,
            int iterations, int begin_save, int save_step, int maxW,
            unsigned long long seed, int verbose=0):
    """Run the ATM Gibbs sampler; return (thetaU, thetaV, lambdaU, lambdaV, pi).

    With ``verbose`` nonzero, prints per-iteration progress with elapsed time and
    an ETA so long runs (minutes of Gibbs) are observable.
    """
    import time as _time
    _t0 = _time.time()
    cdef int M = doc_user.shape[0]
    cdef int S = sent_word_ptr.shape[0] - 1
    cdef int nwords = words.shape[0]

    cdef double alphaSum = K * alpha
    cdef double betaSum = V * beta
    cdef double gammaSum = A * gamma

    # count arrays
    cdef int[:, ::1] nkt = np.zeros((K, V), dtype=np.int32)
    cdef int[::1] nktSum = np.zeros(K, dtype=np.int32)
    cdef int[:, :, ::1] nuak = np.zeros((num_users, A, K), dtype=np.int32)
    cdef int[:, ::1] nua = np.zeros((num_users, A), dtype=np.int32)
    cdef int[:, :, ::1] nvak = np.zeros((num_items, A, K), dtype=np.int32)
    cdef int[:, ::1] nva = np.zeros((num_items, A), dtype=np.int32)
    cdef int[:, ::1] nusa = np.zeros((num_users, A), dtype=np.int32)
    cdef int[:, ::1] nvsa = np.zeros((num_items, A), dtype=np.int32)
    cdef int[::1] nus = np.zeros(num_users, dtype=np.int32)
    cdef int[::1] nvs = np.zeros(num_items, dtype=np.int32)
    cdef int[::1] Ny0 = np.zeros(num_users, dtype=np.int32)
    cdef int[::1] Ny1 = np.zeros(num_users, dtype=np.int32)
    cdef double[::1] NySum = np.zeros(num_users, dtype=np.float64)

    # latent state
    cdef int[::1] y = np.zeros(S, dtype=np.int32)
    cdef int[::1] ya = np.zeros(S, dtype=np.int32)
    cdef int[::1] z = np.zeros(nwords, dtype=np.int32)

    # outputs
    thetaU_np = np.zeros((num_users, A, K), dtype=np.float64)
    thetaV_np = np.zeros((num_items, A, K), dtype=np.float64)
    lambdaU_np = np.zeros((num_users, A), dtype=np.float64)
    lambdaV_np = np.zeros((num_items, A), dtype=np.float64)
    pi_np = np.zeros(num_users, dtype=np.float64)
    cdef double[:, :, ::1] thetaU = thetaU_np
    cdef double[:, :, ::1] thetaV = thetaV_np
    cdef double[:, ::1] lambdaU = lambdaU_np
    cdef double[:, ::1] lambdaV = lambdaV_np
    cdef double[::1] pi = pi_np

    cdef unsigned long long rng = seed if seed != 0 else <unsigned long long>88172645463325252

    # scratch (per sentence)
    cdef double* upw = <double*>malloc(K * sizeof(double))
    cdef double* vpw = <double*>malloc(K * sizeof(double))
    cdef double* upwcum = <double*>malloc(K * sizeof(double))
    cdef double* vpwcum = <double*>malloc(K * sizeof(double))
    cdef double* p = <double*>malloc(2 * A * sizeof(double))
    cdef int* wz_u = <int*>malloc(A * maxW * sizeof(int))
    cdef int* wz_v = <int*>malloc(A * maxW * sizeof(int))

    cdef int m, s, wp, w, W, a, k, t, u, it, initTopic, aspectIdx
    cdef int oldY, oldA, newY, newA, oldTopic, newTopic, wid
    cdef double nyu, nyv, user_aspect_part, item_aspect_part, uSentProb, vSentProb
    cdef double word_part, uat, vat, usum, vsum, rz, mx, add, ra
    cdef int i, i2

    # ---- initialize ----
    for m in range(M):
        u = doc_user[m]
        it = doc_item[m]
        for s in range(doc_sent_ptr[m], doc_sent_ptr[m + 1]):
            aspectIdx = <int>(_rand_double(&rng) * A)
            if aspectIdx == A:
                aspectIdx = A - 1
            ya[s] = aspectIdx
            W = sent_word_ptr[s + 1] - sent_word_ptr[s]
            if _rand_double(&rng) > 0.5:
                y[s] = 0
                Ny0[u] += 1
                nus[u] += 1
                nusa[u][aspectIdx] += 1
            else:
                y[s] = 1
                Ny1[u] += 1
                nvs[it] += 1
                nvsa[it][aspectIdx] += 1
            for wp in range(sent_word_ptr[s], sent_word_ptr[s + 1]):
                wid = words[wp]
                initTopic = <int>(_rand_double(&rng) * K)
                if initTopic == K:
                    initTopic = K - 1
                z[wp] = initTopic
                nkt[initTopic][wid] += 1
                nktSum[initTopic] += 1
                if y[s] == 0:
                    nua[u][aspectIdx] += 1
                    nuak[u][aspectIdx][initTopic] += 1
                else:
                    nva[it][aspectIdx] += 1
                    nvak[it][aspectIdx][initTopic] += 1

    for u in range(num_users):
        NySum[u] = Ny0[u] + Ny1[u] + eta0 + eta1

    # ---- Gibbs iterations ----
    for i in range(iterations):
        if verbose and i > 0:
            _el = _time.time() - _t0
            _eta = _el / i * (iterations - i)
            print("  [ATM] iter %d/%d  elapsed %.0fs  eta %.0fs" % (i, iterations, _el, _eta),
                  flush=True)
        if (i >= begin_save) and (((i - begin_save) % save_step) == 0):
            _estimate(num_users, num_items, A, K, V, alpha, beta, gamma,
                      eta0, eta1, alphaSum, gammaSum,
                      nuak, nua, nvak, nva, nusa, nvsa, nus, nvs, Ny0, NySum,
                      thetaU, thetaV, lambdaU, lambdaV, pi)

        for m in range(M):
            u = doc_user[m]
            it = doc_item[m]
            for s in range(doc_sent_ptr[m], doc_sent_ptr[m + 1]):
                W = sent_word_ptr[s + 1] - sent_word_ptr[s]
                oldY = y[s]
                oldA = ya[s]
                # remove sentence-level counts
                if oldY == 0:
                    Ny0[u] -= 1
                    nus[u] -= 1
                    nusa[u][oldA] -= 1
                    nua[u][oldA] -= W
                else:
                    Ny1[u] -= 1
                    nvs[it] -= 1
                    nvsa[it][oldA] -= 1
                    nva[it][oldA] -= W
                # remove word-topic counts
                for wp in range(sent_word_ptr[s], sent_word_ptr[s + 1]):
                    oldTopic = z[wp]
                    wid = words[wp]
                    nkt[oldTopic][wid] -= 1
                    nktSum[oldTopic] -= 1
                    if oldY == 0:
                        nuak[u][oldA][oldTopic] -= 1
                    else:
                        nvak[it][oldA][oldTopic] -= 1

                nyu = (eta0 + Ny0[u]) / NySum[u]
                nyv = (eta1 + Ny1[u]) / NySum[u]

                for a in range(A):
                    user_aspect_part = nyu * (nusa[u][a] + gamma) / (nus[u] + gammaSum)
                    item_aspect_part = nyv * (nvsa[it][a] + gamma) / (nvs[it] + gammaSum)
                    uSentProb = 0.0
                    vSentProb = 0.0
                    for w in range(W):
                        wid = words[sent_word_ptr[s] + w]
                        usum = 0.0
                        vsum = 0.0
                        for k in range(K):
                            word_part = (nkt[k][wid] + beta) / (nktSum[k] + betaSum)
                            uat = word_part * (nuak[u][a][k] + alpha) / (nua[u][a] + alphaSum)
                            vat = word_part * (nvak[it][a][k] + alpha) / (nva[it][a] + alphaSum)
                            upw[k] = uat
                            vpw[k] = vat
                            usum += uat
                            vsum += vat
                            upwcum[k] = usum
                            vpwcum[k] = vsum
                        rz = _rand_double(&rng) * upwcum[K - 1]
                        for t in range(K):
                            if rz < upwcum[t]:
                                uSentProb += log(upw[t])
                                wz_u[a * maxW + w] = t
                                break
                        rz = _rand_double(&rng) * vpwcum[K - 1]
                        for t in range(K):
                            if rz < vpwcum[t]:
                                vSentProb += log(vpw[t])
                                wz_v[a * maxW + w] = t
                                break
                    mx = -1.0e7
                    if uSentProb > vSentProb:
                        if uSentProb > mx:
                            mx = uSentProb
                    else:
                        if vSentProb > mx:
                            mx = vSentProb
                    add = -mx
                    p[a] = user_aspect_part * exp(uSentProb + add)
                    p[a + A] = item_aspect_part * exp(vSentProb + add)

                for i2 in range(1, 2 * A):
                    p[i2] += p[i2 - 1]
                ra = _rand_double(&rng) * p[2 * A - 1]
                newY = -1
                newA = -1
                for t in range(2 * A):
                    if ra < p[t]:
                        if t < A:
                            newA = t
                            newY = 0
                        else:
                            newA = t - A
                            newY = 1
                        break
                if newA == -1:
                    newA = A - 1
                    newY = 1
                y[s] = newY
                ya[s] = newA

                if newY == 0:
                    Ny0[u] += 1
                    nus[u] += 1
                    nusa[u][newA] += 1
                    nua[u][newA] += W
                else:
                    Ny1[u] += 1
                    nvs[it] += 1
                    nvsa[it][newA] += 1
                    nva[it][newA] += W

                for w in range(W):
                    if newY == 0:
                        newTopic = wz_u[newA * maxW + w]
                    else:
                        newTopic = wz_v[newA * maxW + w]
                    wp = sent_word_ptr[s] + w
                    z[wp] = newTopic
                    wid = words[wp]
                    nkt[newTopic][wid] += 1
                    nktSum[newTopic] += 1
                    if newY == 0:
                        nuak[u][newA][newTopic] += 1
                    else:
                        nvak[it][newA][newTopic] += 1

    # final estimate (faithful: keep the parameters from the last save iteration;
    # if the budget never hit a save iteration, estimate from the final counts).
    if iterations <= begin_save:
        _estimate(num_users, num_items, A, K, V, alpha, beta, gamma,
                  eta0, eta1, alphaSum, gammaSum,
                  nuak, nua, nvak, nva, nusa, nvsa, nus, nvs, Ny0, NySum,
                  thetaU, thetaV, lambdaU, lambdaV, pi)

    free(upw); free(vpw); free(upwcum); free(vpwcum)
    free(p); free(wz_u); free(wz_v)
    return thetaU_np, thetaV_np, lambdaU_np, lambdaV_np, pi_np


cdef void _estimate(int num_users, int num_items, int A, int K, int V,
                    double alpha, double beta, double gamma, double eta0, double eta1,
                    double alphaSum, double gammaSum,
                    int[:, :, ::1] nuak, int[:, ::1] nua,
                    int[:, :, ::1] nvak, int[:, ::1] nva,
                    int[:, ::1] nusa, int[:, ::1] nvsa,
                    int[::1] nus, int[::1] nvs, int[::1] Ny0, double[::1] NySum,
                    double[:, :, ::1] thetaU, double[:, :, ::1] thetaV,
                    double[:, ::1] lambdaU, double[:, ::1] lambdaV,
                    double[::1] pi):
    cdef int u, it, a, k
    for u in range(num_users):
        for a in range(A):
            for k in range(K):
                thetaU[u][a][k] = (nuak[u][a][k] + alpha) / (nua[u][a] + alphaSum)
            lambdaU[u][a] = (nusa[u][a] + gamma) / (nus[u] + gammaSum)
        pi[u] = (eta0 + Ny0[u]) / NySum[u]
    for it in range(num_items):
        for a in range(A):
            for k in range(K):
                thetaV[it][a][k] = (nvak[it][a][k] + alpha) / (nva[it][a] + alphaSum)
            lambdaV[it][a] = (nvsa[it][a] + gamma) / (nvs[it] + gammaSum)


def run_sgd(int num_users, int num_items, int A, int F,
            int[::1] rating_user, int[::1] rating_item, double[::1] ratings,
            double[:, ::1] topic_part,
            double[:, ::1] userFactors, double[:, ::1] itemFactors, double[:, ::1] w,
            double[::1] userBias, double[::1] itemBias,
            double learn_rate, double reg, double weight_reg, double epsilon,
            double mean, int iterations, int verbose):
    """Per-rating SGD, faithful to MFRecommender.trainModel. Updates arrays in place."""
    cdef int N = rating_user.shape[0]
    cdef int e, it_, u, i, a, f
    cdef double err, pred, aspect_rate, tp, uf, vf, weight, dif, ubv, ibv
    cdef double loss, last_loss = 0.0
    cdef int conv_count = 0

    for it_ in range(iterations):
        loss = 0.0
        for e in range(N):
            u = rating_user[e]
            i = rating_item[e]
            # predict
            pred = 0.0
            for a in range(A):
                aspect_rate = 0.0
                for f in range(F):
                    aspect_rate += w[a][f] * w[a][f] * userFactors[u][f] * itemFactors[i][f]
                pred += topic_part[e][a] * aspect_rate
            pred += userBias[u] + itemBias[i] + mean
            err = ratings[e] - pred
            loss += err * err

            ubv = userBias[u]
            userBias[u] += learn_rate * (err - reg * ubv)
            loss += reg * ubv * ubv
            ibv = itemBias[i]
            itemBias[i] += learn_rate * (err - reg * ibv)
            loss += reg * ibv * ibv

            for f in range(F):
                uf = userFactors[u][f]
                vf = itemFactors[i][f]
                dif = 0.0
                for a in range(A):
                    weight = w[a][f]
                    tp = topic_part[e][a]
                    dif += tp * weight * weight
                    loss += weight_reg * fabs(weight)
                    w[a][f] += learn_rate * (err * tp * weight * vf * uf
                                             - weight_reg * 0.5 * weight
                                             * pow(weight * weight + epsilon, -0.5))
                userFactors[u][f] += learn_rate * (err * dif * vf - reg * uf)
                itemFactors[i][f] += learn_rate * (err * dif * uf - reg * vf)
                loss += reg * uf * uf + reg * vf * vf
        loss *= 0.5
        if verbose:
            print("  [ALFM-SGD] iter %d: loss=%.6f" % (it_ + 1, loss))
        # early stopping: |delta loss| < 1e-6 for 3 consecutive iterations
        if it_ > 0:
            if fabs(last_loss - loss) < 1e-6:
                conv_count += 1
                if conv_count >= 3:
                    break
            else:
                conv_count = 0
        last_loss = loss
