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
"""PyTorch network for ALFM.

Ported from the iRev benchmark implementation of

    Cheng, Z., Ding, Y., Zhu, L., & Kankanhalli, M. (2018).
    Aspect-Aware Latent Factor Model: Rating Prediction with Ratings and Reviews.
    WWW 2018.

ALFM represents each user/item review document by a topic (aspect) distribution
obtained from an LDA model (computed in the recommender). The network fuses these
topic vectors with id embeddings, applies an attentive interaction, and predicts
the rating directly. Ported to be self-contained within Cornac.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ALFMModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        topic_dim,
        id_embedding_size=32,
        dropout_rate=0.5,
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, id_embedding_size)
        self.item_embedding = nn.Embedding(num_items, id_embedding_size)

        # Project the LDA topic vector into the id-embedding space (identity-like
        # when topic_dim == id_embedding_size, mirroring iRev's direct addition).
        self.user_topic_fc = nn.Linear(topic_dim, id_embedding_size)
        self.item_topic_fc = nn.Linear(topic_dim, id_embedding_size)

        self.user_fusion = nn.Sequential(
            nn.Linear(id_embedding_size, id_embedding_size), nn.ReLU()
        )
        self.item_fusion = nn.Sequential(
            nn.Linear(id_embedding_size, id_embedding_size), nn.ReLU()
        )

        self.att_layer1 = nn.Sequential(nn.Linear(2 * id_embedding_size, 1), nn.ReLU())
        self.att_layer2 = nn.Linear(1, id_embedding_size, bias=False)

        self.rating_predict = nn.Sequential(
            nn.Linear(id_embedding_size, id_embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(id_embedding_size, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for emb in [self.user_embedding, self.item_embedding]:
            nn.init.xavier_uniform_(emb.weight)
        for fc in [
            self.user_topic_fc,
            self.item_topic_fc,
            self.user_fusion[0],
            self.item_fusion[0],
            self.att_layer1[0],
            self.att_layer2,
            self.rating_predict[0],
            self.rating_predict[3],
        ]:
            nn.init.uniform_(fc.weight, -0.1, 0.1)
            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0.1)

    def forward(self, uids, iids, user_topic, item_topic):
        user_id_embed = self.user_embedding(uids)
        item_id_embed = self.item_embedding(iids)

        user_embed = user_id_embed + self.user_topic_fc(user_topic)
        item_embed = item_id_embed + self.item_topic_fc(item_topic)
        user_embed = self.user_fusion(user_embed)
        item_embed = self.item_fusion(item_embed)

        feature_all = torch.cat((user_embed, item_embed), dim=-1)
        att_weights = self.att_layer2(self.att_layer1(feature_all))
        att_weights = F.softmax(att_weights, dim=-1)

        interact = att_weights * user_embed * item_embed
        prediction = self.rating_predict(interact)
        return prediction.squeeze(1)
