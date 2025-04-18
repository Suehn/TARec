import os
import numpy as np
from time import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.functions import NormLayer


class Student_LightGCL(nn.Module):
    def __init__(
        self,
        adj,
        n_users,
        n_items,
        embedding_dim,
        gnn_layer,
        ut,
        vt,
        u_mul_s,
        v_mul_s,
        args,
    ):
        super().__init__()

        self.args = args
        self.adj = adj
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.gnn_layer = gnn_layer

        self.ut = ut
        self.vt = vt
        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s

        self.user_id_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.user_id_embedding_pre = nn.Embedding(n_users, embedding_dim)
        self.item_id_embedding_pre = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_id_embedding_pre.weight)
        nn.init.xavier_uniform_(self.item_id_embedding_pre.weight)
        self.user_id_embedding_pre.weight.requires_grad = False
        self.item_id_embedding_pre.weight.requires_grad = False

        self.norm_layer = NormLayer(args.ta_norm_mode, args.ta_norm_scale)

    def forward(
        self,
        adj_norm,
        image_item_embeds,
        text_item_embeds,
        image_user_embeds,
        text_user_embeds,
        users=None,
        pos_items=None,
        neg_items=None,
        is_test=False,
    ):
        args = self.args

        user_embedding = self.user_id_embedding.weight
        item_embedding = self.item_id_embedding.weight

        combined_embeddings = torch.cat((user_embedding, item_embedding), dim=0)

        norm_embeddings = self.norm_layer(combined_embeddings, self.adj)
        user_embedding, item_embedding = torch.split(
            norm_embeddings, [self.n_users, self.n_items], dim=0
        )

        if is_test:
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            #
            # start.record()
            gnn_layer = self.gnn_layer
            self.E_u_list = [None] * (gnn_layer + 1)
            self.E_i_list = [None] * (gnn_layer + 1)
            self.Z_u_list = [None] * (gnn_layer + 1)
            self.Z_i_list = [None] * (gnn_layer + 1)

            self.E_u_list[0] = user_embedding
            self.E_i_list[0] = item_embedding

            for layer in range(1, self.gnn_layer + 1):

                self.Z_u_list[layer] = torch.sparse.mm(
                    adj_norm, self.E_i_list[layer - 1]
                )
                self.Z_i_list[layer] = torch.sparse.mm(
                    adj_norm.T, self.E_u_list[layer - 1]
                )

                self.E_u_list[layer] = self.Z_u_list[layer]
                self.E_i_list[layer] = self.Z_i_list[layer]

            user_embedding = torch.stack(self.E_u_list, dim=1).mean(
                dim=1, keepdim=False
            )
            item_embedding = torch.stack(self.E_i_list, dim=1).mean(
                dim=1, keepdim=False
            )

            user_embedding = user_embedding
            item_embedding = item_embedding
            # end.record()
            # torch.cuda.synchronize()
            # print(f"GNN forward time: {start.elapsed_time(end) / 1000} s")
            #
            return user_embedding, item_embedding

        else:

            gcl_loss = self.cal_gcl_loss(
                adj_norm=adj_norm, users=users, pos_items=pos_items, neg_items=neg_items
            )

            user_embedding = self.E_u
            item_embedding = self.E_i

            user_embedding = user_embedding
            item_embedding = item_embedding

            return user_embedding, item_embedding, gcl_loss

    def init_user_item_embed(
        self, user_pretrained_embedding, item_pretrained_embedding
    ):
        self.user_id_embedding = nn.Embedding.from_pretrained(
            user_pretrained_embedding, freeze=False
        )
        self.item_id_embedding = nn.Embedding.from_pretrained(
            item_pretrained_embedding, freeze=False
        )
        self.user_id_embedding_pre = nn.Embedding.from_pretrained(
            user_pretrained_embedding, freeze=True
        )
        self.item_id_embedding_pre = nn.Embedding.from_pretrained(
            item_pretrained_embedding, freeze=True
        )

    def cal_gcl_loss(self, adj_norm, users, pos_items, neg_items):

        user_embedding = self.user_id_embedding.weight
        item_embedding = self.item_id_embedding.weight
        t_user_embedding = self.user_id_embedding_pre.weight
        t_item_embedding = self.item_id_embedding_pre.weight

        combined_embeddings = torch.cat((user_embedding, item_embedding), dim=0)

        norm_embeddings = self.norm_layer(combined_embeddings, self.adj)
        user_embedding, item_embedding = torch.split(
            norm_embeddings, [self.n_users, self.n_items], dim=0
        )

        gnn_layer = self.gnn_layer
        self.E_u_list = [None] * (gnn_layer + 1)
        self.E_i_list = [None] * (gnn_layer + 1)
        self.Z_u_list = [None] * (gnn_layer + 1)
        self.Z_i_list = [None] * (gnn_layer + 1)
        self.G_u_list = [None] * (gnn_layer + 1)
        self.G_i_list = [None] * (gnn_layer + 1)

        self.X_u_list = [None] * (gnn_layer + 1)
        self.X_i_list = [None] * (gnn_layer + 1)

        self.XG_u_list = [None] * (gnn_layer + 1)
        self.XG_i_list = [None] * (gnn_layer + 1)

        self.TE_i_list = [None] * (gnn_layer + 1)
        self.TE_u_list = [None] * (gnn_layer + 1)

        uids = torch.tensor(users).long().cuda()
        iids = torch.tensor(pos_items).long().cuda()

        self.E_u_list[0] = self.G_u_list[0] = self.XG_u_list[0] = user_embedding
        self.E_i_list[0] = self.G_i_list[0] = self.XG_i_list[0] = item_embedding
        self.TE_u_list[0] = t_user_embedding
        self.TE_i_list[0] = t_item_embedding

        for layer in range(1, self.gnn_layer + 1):

            self.Z_u_list[layer] = torch.sparse.mm(adj_norm, self.E_i_list[layer - 1])
            self.Z_i_list[layer] = torch.sparse.mm(adj_norm.T, self.E_u_list[layer - 1])

            self.TE_u_list[layer] = torch.sparse.mm(adj_norm, self.TE_i_list[layer - 1])
            self.TE_i_list[layer] = torch.sparse.mm(
                adj_norm.T, self.TE_u_list[layer - 1]
            )

            vt_ei = self.vt @ self.E_i_list[layer - 1]
            self.G_u_list[layer] = self.u_mul_s @ vt_ei
            ut_eu = self.ut @ self.E_u_list[layer - 1]
            self.G_i_list[layer] = self.v_mul_s @ ut_eu

            u_random_noise = torch.rand_like(self.E_u_list[layer - 1]).cuda()
            i_random_noise = torch.rand_like(self.E_i_list[layer - 1]).cuda()
            self.X_u_list[layer] = (
                self.Z_u_list[layer]
                + torch.sign(self.Z_u_list[layer])
                * F.normalize(u_random_noise, p=2, dim=1)
                * self.args.eps
            )
            self.X_i_list[layer] = (
                self.Z_i_list[layer]
                + torch.sign(self.Z_i_list[layer])
                * F.normalize(i_random_noise, p=2, dim=1)
                * self.args.eps
            )

            self.XG_u_list[layer] = (
                self.G_u_list[layer]
                + torch.sign(self.G_u_list[layer])
                * F.normalize(u_random_noise, p=2, dim=1)
                * self.args.eps
            )
            self.XG_i_list[layer] = (
                self.G_i_list[layer]
                + torch.sign(self.G_i_list[layer])
                * F.normalize(i_random_noise, p=2, dim=1)
                * self.args.eps
            )

            self.E_u_list[layer] = self.Z_u_list[layer]
            self.E_i_list[layer] = self.Z_i_list[layer]

        self.E_u = torch.stack(self.E_u_list, dim=1).mean(dim=1, keepdim=False)
        self.E_i = torch.stack(self.E_i_list, dim=1).mean(dim=1, keepdim=False)

        self.G_u = torch.stack(self.G_u_list, dim=1).mean(dim=1, keepdim=False)
        self.G_i = torch.stack(self.G_i_list, dim=1).mean(dim=1, keepdim=False)

        self.X_u = torch.stack(self.X_u_list[-1:], dim=1).mean(dim=1, keepdim=False)
        self.X_i = torch.stack(self.X_i_list[-1:], dim=1).mean(dim=1, keepdim=False)

        self.XG_u = torch.stack(self.XG_u_list, dim=1).mean(dim=1, keepdim=False)
        self.XG_i = torch.stack(self.XG_i_list, dim=1).mean(dim=1, keepdim=False)

        neg_score = torch.logsumexp(
            (self.G_u[uids] @ self.E_u[uids].T) / 0.2, dim=1
        ).mean()
        neg_score += torch.logsumexp(
            (self.G_i[iids] @ self.E_i[iids].T) / 0.2, dim=1
        ).mean()
        neg_score += torch.logsumexp(
            (self.X_u[uids] @ self.E_u[uids].T) / 0.15, dim=1
        ).mean()
        neg_score += torch.logsumexp(
            (self.X_i[iids] @ self.E_i[iids].T) / 0.15, dim=1
        ).mean()

        pos_score = (
            (
                torch.clamp(
                    (self.G_u[uids] * self.E_u[uids]).sum(dim=1) / 0.2, -5.0, 5.0
                )
            ).mean()
            + (
                torch.clamp(
                    (self.G_i[iids] * self.E_i[iids]).sum(dim=1) / 0.2, -5.0, 5.0
                )
            ).mean()
            + torch.clamp(
                (self.X_u[uids] * self.E_u[uids]).sum(dim=1) / 0.15, -5.0, 5.0
            ).mean()
            + torch.clamp(
                (self.X_i[iids] * self.E_i[iids]).sum(dim=1) / 0.15, -5.0, 5.0
            ).mean()
        )
        gcl_loss = -pos_score + neg_score
        gcl_loss *= len(self.G_u_list) + 1

        layer_neg_score = (
            torch.logsumexp(
                (self.E_u_list[0][uids] @ self.TE_u_list[1][pos_items].T) / 0.2, dim=1
            ).mean()
            + torch.logsumexp(
                (self.TE_u_list[1][uids] @ self.E_i_list[0][pos_items].T) / 0.2, dim=1
            ).mean()
        )
        layer_neg_score += (
            torch.logsumexp(
                self.E_u_list[0][uids] @ self.TE_u_list[1][uids].T / 0.2, dim=1
            ).mean()
            + torch.logsumexp(
                self.E_i_list[0][iids] @ self.TE_i_list[1][iids].T / 0.2, dim=1
            ).mean()
        )
        layer_pos_score = torch.clamp(
            (self.E_u_list[0][uids] * self.TE_i_list[1][pos_items]).sum(dim=1) / 0.2,
            -5.0,
            5.0,
        ).mean()
        +torch.clamp(
            (self.TE_u_list[1][uids] * self.E_i_list[0][pos_items]).sum(dim=1) / 0.2,
            -5.0,
            5.0,
        ).mean()
        +torch.clamp(
            (self.E_u_list[0][uids] * self.TE_u_list[1][uids]).sum(dim=1) / 0.2,
            -5.0,
            5.0,
        ).mean()
        +torch.clamp(
            (self.E_i_list[0][iids] * self.TE_i_list[1][iids]).sum(dim=1) / 0.2,
            -5.0,
            5.0,
        ).mean()
        layer_neg_score = 50 * layer_neg_score
        layer_pos_score = 50 * layer_pos_score

        layer_neg_score += (
            torch.logsumexp(
                (self.E_u_list[0][uids] @ self.E_u_list[1][pos_items].T) / 0.2, dim=1
            ).mean()
            + torch.logsumexp(
                (self.E_u_list[1][uids] @ self.E_i_list[0][pos_items].T) / 0.2, dim=1
            ).mean()
        )
        layer_neg_score += (
            torch.logsumexp(
                self.E_u_list[0][uids] @ self.E_u_list[1][uids].T / 0.2, dim=1
            ).mean()
            + torch.logsumexp(
                self.E_i_list[0][iids] @ self.E_i_list[1][iids].T / 0.2, dim=1
            ).mean()
        )
        layer_pos_score += torch.clamp(
            (self.E_u_list[0][uids] * self.E_i_list[1][pos_items]).sum(dim=1) / 0.2,
            -5.0,
            5.0,
        ).mean()
        +torch.clamp(
            (self.E_u_list[1][uids] * self.E_i_list[0][pos_items]).sum(dim=1) / 0.2,
            -5.0,
            5.0,
        ).mean()
        +torch.clamp(
            (self.E_u_list[0][uids] * self.E_u_list[1][uids]).sum(dim=1) / 0.2,
            -5.0,
            5.0,
        ).mean()
        +torch.clamp(
            (self.E_i_list[0][iids] * self.E_i_list[1][iids]).sum(dim=1) / 0.2,
            -5.0,
            5.0,
        ).mean()

        layer_loss = -layer_pos_score + layer_neg_score
        gcl_loss += layer_loss

        return gcl_loss

    def normalize(self, input, p=2, dim=1, eps=1e-12):

        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input)

        norm = torch.norm(input, p, dim, keepdim=True).clamp_min(eps).expand_as(input)
        return input / norm

    def InfoNCE(self, view1, view2, temperature: float = 0.15, b_cos: bool = True):
        """
        Args:
            view1: (torch.Tensor - N x D)
            view2: (torch.Tensor - N x D)
            temperature: float
            b_cos (bool)

        Return: Average InfoNCE Loss
        """
        if b_cos:
            view1, view2 = self.normalize(view1, dim=1), self.normalize(view2, dim=1)

        pos_score = (view1 @ view2.T) / temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()
