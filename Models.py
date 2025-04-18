import os
import numpy as np
from time import time
import pickle
import pickle
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from sklearn.decomposition import PCA, FastICA
from sklearn import manifold


from utility.parser import args


class Teacher_Model(nn.Module):
    def __init__(
        self, n_users, n_items, embedding_dim, gnn_layer, image_feats, text_feats
    ):
        super(Teacher_Model, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_ui_layers = self.gnn_layer = gnn_layer

        self.user_id_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_id_embedding.weight)
        nn.init.xavier_normal_(self.item_id_embedding.weight)

        self.image_feats = torch.tensor(image_feats).float().cuda()
        self.text_feats = torch.tensor(text_feats).float().cuda()
        self.image_embedding = nn.Embedding.from_pretrained(
            torch.Tensor(image_feats), freeze=False
        )
        self.text_embedding = nn.Embedding.from_pretrained(
            torch.Tensor(text_feats), freeze=False
        )

        self.image_trans = nn.Linear(self.image_feats.shape[1], self.embedding_dim)
        self.text_trans = nn.Linear(self.text_feats.shape[1], self.embedding_dim)
        nn.init.xavier_uniform_(self.image_trans.weight)
        nn.init.xavier_uniform_(self.text_trans.weight)
        self.encoder = nn.ModuleDict(
            {"image_encoder": self.image_trans, "text_encoder": self.text_trans}
        )

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.batch_norm = nn.BatchNorm1d(self.embedding_dim)

    def mm(self, x, y):
        return torch.sparse.mm(x, y) if args.sparse else torch.mm(x, y)

    def forward(self, ui_graph, iu_graph, prompt_module=None):

        if prompt_module is None:
            prompt_user = torch.zeros((self.n_users, self.embedding_dim)).cuda()
            prompt_item = torch.zeros((self.n_items, self.embedding_dim)).cuda()
        else:
            prompt_user, prompt_item = prompt_module()

        feat_prompt_item_image = torch.mm(
            prompt_item, torch.mm(prompt_item.T, self.image_feats)
        )
        feat_prompt_item_text = torch.mm(
            prompt_item, torch.mm(prompt_item.T, self.text_feats)
        )

        image_feat = self.dropout(
            self.image_trans(
                self.image_feats
                + args.feat_soft_token_rate
                * F.normalize(feat_prompt_item_image, p=2, dim=1)
            )
        )
        text_feat = self.dropout(
            self.text_trans(
                self.text_feats
                + args.feat_soft_token_rate
                * F.normalize(feat_prompt_item_text, p=2, dim=1)
            )
        )

        for _ in range(self.gnn_layer):
            image_user_feats = self.mm(ui_graph, image_feat)
            image_item_feats = self.mm(iu_graph, image_user_feats)

            text_user_feats = self.mm(ui_graph, text_feat)
            text_item_feats = self.mm(iu_graph, text_user_feats)

        u_g_embeddings = (
            self.user_id_embedding.weight
            + args.soft_token_rate * F.normalize(prompt_user, p=2, dim=1)
        )
        i_g_embeddings = (
            self.item_id_embedding.weight
            + args.soft_token_rate * F.normalize(prompt_item, p=2, dim=1)
        )

        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]

        for i in range(self.gnn_layer):

            if i == (self.gnn_layer - 1):
                u_g_embeddings = self.softmax(torch.mm(ui_graph, i_g_embeddings))
                i_g_embeddings = self.softmax(torch.mm(iu_graph, u_g_embeddings))
            else:
                u_g_embeddings = torch.mm(ui_graph, i_g_embeddings)
                i_g_embeddings = torch.mm(iu_graph, u_g_embeddings)

            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        user_embeddings = torch.stack(user_emb_list, dim=0).mean(dim=0)
        item_embeddings = torch.stack(item_emb_list, dim=0).mean(dim=0)

        u_g_embeddings = (
            user_embeddings
            + args.model_cat_rate * F.normalize(image_user_feats, p=2, dim=1)
            + args.model_cat_rate * F.normalize(text_user_feats, p=2, dim=1)
        )
        i_g_embeddings = (
            item_embeddings
            + args.model_cat_rate * F.normalize(image_item_feats, p=2, dim=1)
            + args.model_cat_rate * F.normalize(text_item_feats, p=2, dim=1)
        )

        return (
            u_g_embeddings,
            i_g_embeddings,
            image_item_feats,
            text_item_feats,
            image_user_feats,
            text_user_feats,
            u_g_embeddings,
            i_g_embeddings,
            prompt_user,
            prompt_item,
        )


class Teacher_Model_XGCL(nn.Module):
    def __init__(
        self, n_users, n_items, embedding_dim, gnn_layer, image_feats, text_feats
    ):
        super(Teacher_Model_XGCL, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_ui_layers = self.gnn_layer = gnn_layer

        self.user_id_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_id_embedding.weight)
        nn.init.xavier_normal_(self.item_id_embedding.weight)

        self.image_feats = torch.tensor(image_feats).float().cuda()
        self.text_feats = torch.tensor(text_feats).float().cuda()
        self.image_embedding = nn.Embedding.from_pretrained(
            torch.Tensor(image_feats), freeze=False
        )
        self.text_embedding = nn.Embedding.from_pretrained(
            torch.Tensor(text_feats), freeze=False
        )

        self.image_trans = nn.Linear(self.image_feats.shape[1], self.embedding_dim)
        self.text_trans = nn.Linear(self.text_feats.shape[1], self.embedding_dim)
        nn.init.xavier_uniform_(self.image_trans.weight)
        nn.init.xavier_uniform_(self.text_trans.weight)
        self.encoder = nn.ModuleDict(
            {"image_encoder": self.image_trans, "text_encoder": self.text_trans}
        )

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.batch_norm = nn.BatchNorm1d(self.embedding_dim)

    def mm(self, x, y):
        return torch.sparse.mm(x, y) if args.sparse else torch.mm(x, y)

    def forward(
        self,
        ui_graph,
        iu_graph,
        prompt_module=None,
        users=None,
        pos_items=None,
        neg_items=None,
    ):

        if prompt_module is None:
            prompt_user = torch.zeros((self.n_users, self.embedding_dim)).cuda()
            prompt_item = torch.zeros((self.n_items, self.embedding_dim)).cuda()
        else:
            prompt_user, prompt_item = prompt_module()

        feat_prompt_item_image = torch.mm(
            prompt_item, torch.mm(prompt_item.T, self.image_feats)
        )
        feat_prompt_item_text = torch.mm(
            prompt_item, torch.mm(prompt_item.T, self.text_feats)
        )

        image_feat = self.dropout(
            self.image_trans(
                self.image_feats
                + args.feat_soft_token_rate
                * F.normalize(feat_prompt_item_image, p=2, dim=1)
            )
        )
        text_feat = self.dropout(
            self.text_trans(
                self.text_feats
                + args.feat_soft_token_rate
                * F.normalize(feat_prompt_item_text, p=2, dim=1)
            )
        )

        for _ in range(self.gnn_layer):
            image_user_feats = self.mm(ui_graph, image_feat)
            image_item_feats = self.mm(iu_graph, image_user_feats)

            text_user_feats = self.mm(ui_graph, text_feat)
            text_item_feats = self.mm(iu_graph, text_user_feats)

        u_g_embeddings = (
            self.user_id_embedding.weight
            + args.soft_token_rate * F.normalize(prompt_user, p=2, dim=1)
        )
        i_g_embeddings = (
            self.item_id_embedding.weight
            + args.soft_token_rate * F.normalize(prompt_item, p=2, dim=1)
        )

        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]

        for i in range(self.gnn_layer):
            if i == (self.gnn_layer - 1):
                u_g_embeddings = torch.mm(ui_graph, i_g_embeddings)
                i_g_embeddings = torch.mm(iu_graph, u_g_embeddings)

                u_g_embeddings_cl = u_g_embeddings
                i_g_embeddings_cl = i_g_embeddings
            else:
                u_g_embeddings = torch.mm(ui_graph, i_g_embeddings)
                i_g_embeddings = torch.mm(iu_graph, u_g_embeddings)

                u_random_noise = torch.rand_like(u_g_embeddings).cuda()
                i_random_noise = torch.rand_like(i_g_embeddings).cuda()

                u_g_embeddings += (
                    torch.sign(u_g_embeddings)
                    * F.normalize(u_random_noise, p=2, dim=-1)
                    * args.eps
                )
                i_g_embeddings += (
                    torch.sign(i_g_embeddings)
                    * F.normalize(i_random_noise, p=2, dim=-1)
                    * args.eps
                )

            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        user_embeddings = torch.stack(user_emb_list, dim=0).mean(dim=0)
        item_embeddings = torch.stack(item_emb_list, dim=0).mean(dim=0)

        u_g_embeddings = (
            user_embeddings
            + args.model_cat_rate * F.normalize(image_user_feats, p=2, dim=1)
            + args.model_cat_rate * F.normalize(text_user_feats, p=2, dim=1)
        )
        i_g_embeddings = (
            item_embeddings
            + args.model_cat_rate * F.normalize(image_item_feats, p=2, dim=1)
            + args.model_cat_rate * F.normalize(text_item_feats, p=2, dim=1)
        )

        u_g_embeddings_cl = (
            u_g_embeddings_cl
            + args.model_cat_rate * F.normalize(image_user_feats, p=2, dim=1)
            + args.model_cat_rate * F.normalize(text_user_feats, p=2, dim=1)
        )
        i_g_embeddings_cl = (
            i_g_embeddings_cl
            + args.model_cat_rate * F.normalize(image_item_feats, p=2, dim=1)
            + args.model_cat_rate * F.normalize(text_item_feats, p=2, dim=1)
        )

        if users is not None and round(args.t_cl_loss_rate, 9) != 0.0:

            uids = torch.tensor(users).long().cuda()
            iids = torch.tensor(pos_items + neg_items).long().cuda()

            gcl_loss = self.cal_xgcl_loss(
                user_embeddings[uids],
                item_embeddings[iids],
                u_g_embeddings_cl[uids],
                i_g_embeddings_cl[iids],
            )
        else:
            gcl_loss = 0.0

        return (
            u_g_embeddings,
            i_g_embeddings,
            image_item_feats,
            text_item_feats,
            image_user_feats,
            text_user_feats,
            u_g_embeddings,
            i_g_embeddings,
            prompt_user,
            prompt_item,
            gcl_loss,
        )

    def cal_xgcl_loss(
        self, user_embeddings, item_embeddings, user_embeddings_cl, item_embeddings_cl
    ):
        user_cl_loss = self.InfoNCE(user_embeddings, user_embeddings_cl)
        item_cl_loss = self.InfoNCE(item_embeddings, item_embeddings_cl)
        return user_cl_loss + item_cl_loss

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


class Teacher_Model_GCL(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim,
        gnn_layer,
        image_feats,
        text_feats,
        ut,
        vt,
        u_mul_s,
        v_mul_s,
    ):
        super(Teacher_Model_GCL, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_ui_layers = self.gnn_layer = gnn_layer

        self.user_id_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_id_embedding.weight)
        nn.init.xavier_normal_(self.item_id_embedding.weight)

        self.image_feats = torch.tensor(image_feats).float().cuda()
        self.text_feats = torch.tensor(text_feats).float().cuda()
        self.image_embedding = nn.Embedding.from_pretrained(
            torch.Tensor(image_feats), freeze=False
        )
        self.text_embedding = nn.Embedding.from_pretrained(
            torch.Tensor(text_feats), freeze=False
        )

        self.image_trans = nn.Linear(self.image_feats.shape[1], self.embedding_dim)
        self.text_trans = nn.Linear(self.text_feats.shape[1], self.embedding_dim)
        nn.init.xavier_uniform_(self.image_trans.weight)
        nn.init.xavier_uniform_(self.text_trans.weight)
        self.encoder = nn.ModuleDict(
            {"image_encoder": self.image_trans, "text_encoder": self.text_trans}
        )

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.batch_norm = nn.BatchNorm1d(self.embedding_dim)

        self.vt = vt
        self.ut = ut
        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.gnn_layer = gnn_layer

    def mm(self, x, y):
        return torch.sparse.mm(x, y) if args.sparse else torch.mm(x, y)

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

    def forward(
        self,
        ui_graph,
        iu_graph,
        prompt_module=None,
        users=None,
        pos_items=None,
        neg_items=None,
    ):
        if users is not None and (round(args.t_cl_loss_rate, 9) != 0):

            gcl_loss = self.cal_gcl_loss(
                adj_norm=ui_graph,
                u_g_embeddings=self.user_id_embedding.weight,
                i_g_embeddings=self.item_id_embedding.weight,
                users=users,
                pos_items=pos_items,
                neg_items=neg_items,
            )

        else:
            gcl_loss = 0.0

        if prompt_module is None:
            prompt_user = torch.zeros((self.n_users, self.embedding_dim)).cuda()
            prompt_item = torch.zeros((self.n_items, self.embedding_dim)).cuda()
        else:
            prompt_user, prompt_item = prompt_module()

        feat_prompt_item_image = torch.mm(
            prompt_item, torch.mm(prompt_item.T, self.image_feats)
        )
        feat_prompt_item_text = torch.mm(
            prompt_item, torch.mm(prompt_item.T, self.text_feats)
        )

        image_feat = self.dropout(
            self.image_trans(
                self.image_feats
                + args.feat_soft_token_rate
                * F.normalize(feat_prompt_item_image, p=2, dim=1)
            )
        )
        text_feat = self.dropout(
            self.text_trans(
                self.text_feats
                + args.feat_soft_token_rate
                * F.normalize(feat_prompt_item_text, p=2, dim=1)
            )
        )

        for _ in range(self.gnn_layer):
            image_user_feats = self.mm(ui_graph, image_feat)
            image_item_feats = self.mm(iu_graph, image_user_feats)

            text_user_feats = self.mm(ui_graph, text_feat)
            text_item_feats = self.mm(iu_graph, text_user_feats)

        u_g_embeddings = (
            self.user_id_embedding.weight
            + args.soft_token_rate * F.normalize(prompt_user, p=2, dim=1)
        )
        i_g_embeddings = (
            self.item_id_embedding.weight
            + args.soft_token_rate * F.normalize(prompt_item, p=2, dim=1)
        )

        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]

        for i in range(self.gnn_layer):
            if i == (self.gnn_layer - 1):
                u_g_embeddings = self.softmax(torch.mm(ui_graph, i_g_embeddings))
                i_g_embeddings = self.softmax(torch.mm(iu_graph, u_g_embeddings))
            else:
                u_g_embeddings = torch.mm(ui_graph, i_g_embeddings)
                i_g_embeddings = torch.mm(iu_graph, u_g_embeddings)

            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        user_embeddings = torch.stack(user_emb_list, dim=0).mean(dim=0)
        item_embeddings = torch.stack(item_emb_list, dim=0).mean(dim=0)

        u_g_embeddings = (
            user_embeddings
            + args.model_cat_rate * F.normalize(image_user_feats, p=2, dim=1)
            + args.model_cat_rate * F.normalize(text_user_feats, p=2, dim=1)
        )
        i_g_embeddings = (
            item_embeddings
            + args.model_cat_rate * F.normalize(image_item_feats, p=2, dim=1)
            + args.model_cat_rate * F.normalize(text_item_feats, p=2, dim=1)
        )

        return (
            u_g_embeddings,
            i_g_embeddings,
            image_item_feats,
            text_item_feats,
            image_user_feats,
            text_user_feats,
            u_g_embeddings,
            i_g_embeddings,
            prompt_user,
            prompt_item,
            gcl_loss,
        )

    def cal_gcl_loss(
        self, adj_norm, u_g_embeddings, i_g_embeddings, users, pos_items, neg_items
    ):
        user_embedding = u_g_embeddings
        item_embedding = i_g_embeddings

        gnn_layer = self.gnn_layer
        self.E_u_list = [None] * (gnn_layer + 1)
        self.E_i_list = [None] * (gnn_layer + 1)
        self.Z_u_list = [None] * (gnn_layer + 1)
        self.Z_i_list = [None] * (gnn_layer + 1)
        self.G_u_list = [None] * (gnn_layer + 1)
        self.G_i_list = [None] * (gnn_layer + 1)

        uids = torch.tensor(users).long().cuda()
        iids = torch.tensor(pos_items + neg_items).long().cuda()

        self.E_u_list[0] = self.G_u_list[0] = user_embedding
        self.E_i_list[0] = self.G_i_list[0] = item_embedding

        for layer in range(1, self.gnn_layer + 1):

            self.Z_u_list[layer] = torch.sparse.mm(adj_norm, self.E_i_list[layer - 1])
            self.Z_i_list[layer] = torch.sparse.mm(adj_norm.T, self.E_u_list[layer - 1])

            vt_ei = self.vt @ self.E_i_list[layer - 1]
            self.G_u_list[layer] = self.u_mul_s @ vt_ei
            ut_eu = self.ut @ self.E_u_list[layer - 1]
            self.G_i_list[layer] = self.v_mul_s @ ut_eu

            self.E_u_list[layer] = self.Z_u_list[layer]
            self.E_i_list[layer] = self.Z_i_list[layer]

        self.G_u = torch.stack(self.G_u_list, dim=1).mean(dim=1, keepdim=False)
        self.G_i = torch.stack(self.G_i_list, dim=1).mean(dim=1, keepdim=False)

        self.E_u = torch.stack(self.E_u_list, dim=1).mean(dim=1, keepdim=False)
        self.E_i = torch.stack(self.E_i_list, dim=1).mean(dim=1, keepdim=False)

        neg_score = torch.logsumexp((self.G_u[uids] @ self.E_u.T) / 0.2, dim=1).mean()
        neg_score += torch.logsumexp((self.G_i[iids] @ self.E_i.T) / 0.2, dim=1).mean()
        pos_score = (
            torch.clamp((self.G_u[uids] * self.E_u[uids]).sum(dim=1) / 0.2, -5.0, 5.0)
        ).mean() + (
            torch.clamp((self.G_i[iids] * self.E_i[iids]).sum(dim=1) / 0.2, -5.0, 5.0)
        ).mean()
        gcl_loss = -pos_score + neg_score
        gcl_loss *= 0.01 * (len(self.G_u_list) + 1)
        gcl_loss_user = self.InfoNCE(self.G_u[uids], self.E_u[uids])
        gcl_loss_item = self.InfoNCE(self.G_i[iids], self.E_i[iids])
        gcl_loss += gcl_loss_user + gcl_loss_item

        return gcl_loss


class PromptLearner(nn.Module):
    def __init__(self, image_feats=None, text_feats=None, ui_graph=None):
        super().__init__()
        self.ui_graph = ui_graph

        if args.hard_token_type == "pca":
            try:
                t1 = time()
                hard_token_image = pickle.load(
                    open(
                        os.path.join(args.data_path, args.dataset)
                        + f"/hard_token_image_pca_{args.embed_size}",
                        "rb",
                    )
                )
                hard_token_text = pickle.load(
                    open(
                        os.path.join(args.data_path, args.dataset)
                        + f"/hard_token_text_pca_{args.embed_size}",
                        "rb",
                    )
                )
                print("already load hard token", time() - t1)
            except Exception:
                hard_token_image = PCA(n_components=args.embed_size).fit_transform(
                    image_feats
                )
                hard_token_text = PCA(n_components=args.embed_size).fit_transform(
                    text_feats
                )
                pickle.dump(
                    hard_token_image,
                    open(
                        os.path.join(args.data_path, args.dataset)
                        + f"/hard_token_image_pca_{args.embed_size}",
                        "wb",
                    ),
                )
                pickle.dump(
                    hard_token_text,
                    open(
                        os.path.join(args.data_path, args.dataset)
                        + f"/hard_token_text_pca_{args.embed_size}",
                        "wb",
                    ),
                )
        elif args.hard_token_type == "ica":
            try:
                t1 = time()
                hard_token_image = pickle.load(
                    open(
                        os.path.join(args.data_path, args.dataset)
                        + "/hard_token_image_ica",
                        "rb",
                    )
                )
                hard_token_text = pickle.load(
                    open(
                        os.path.join(args.data_path, args.dataset)
                        + "/hard_token_text_ica",
                        "rb",
                    )
                )
                print("already load hard token", time() - t1)
            except Exception:
                hard_token_image = FastICA(
                    n_components=args.embed_size, random_state=12
                ).fit_transform(image_feats)
                hard_token_text = FastICA(
                    n_components=args.embed_size, random_state=12
                ).fit_transform(text_feats)
                pickle.dump(
                    hard_token_image,
                    open(
                        os.path.join(args.data_path, args.dataset)
                        + "/hard_token_image_ica",
                        "wb",
                    ),
                )
                pickle.dump(
                    hard_token_text,
                    open(
                        os.path.join(args.data_path, args.dataset)
                        + "/hard_token_text_ica",
                        "wb",
                    ),
                )
        elif args.hard_token_type == "isomap":
            hard_token_image = manifold.Isomap(
                n_neighbors=5, n_components=args.embed_size, n_jobs=-1
            ).fit_transform(image_feats)
            hard_token_text = manifold.Isomap(
                n_neighbors=5, n_components=args.embed_size, n_jobs=-1
            ).fit_transform(text_feats)

        self.item_hard_token = torch.mean(
            (
                torch.stack(
                    (
                        torch.tensor(hard_token_image).float(),
                        torch.tensor(hard_token_text).float(),
                    )
                )
            ),
            dim=0,
        ).cuda()
        self.user_hard_token = torch.mm(ui_graph, self.item_hard_token).cuda()

        self.trans_user = nn.Linear(args.embed_size, args.embed_size).cuda()
        self.trans_item = nn.Linear(args.embed_size, args.embed_size).cuda()

    def forward(self):

        return F.dropout(
            self.trans_user(self.user_hard_token), args.prompt_dropout
        ), F.dropout(self.trans_item(self.item_hard_token), args.prompt_dropout)


class Student_LightGCL(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim,
        gnn_layer,
        ut,
        vt,
        u_mul_s,
        v_mul_s,
        image_feats=None,
        text_feats=None,
    ):
        super().__init__()
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

        user_embedding = (
            self.user_id_embedding_pre.weight + self.user_id_embedding.weight
        )
        item_embedding = (
            self.item_id_embedding_pre.weight + self.item_id_embedding.weight
        )

        if is_test:
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

            user_embedding = (
                user_embedding
                + args.model_cat_rate * F.normalize(image_user_embeds, p=2, dim=1)
                + args.model_cat_rate * F.normalize(text_user_embeds, p=2, dim=1)
            )
            item_embedding = (
                item_embedding
                + args.model_cat_rate * F.normalize(image_item_embeds, p=2, dim=1)
                + args.model_cat_rate * F.normalize(text_item_embeds, p=2, dim=1)
            )

            return user_embedding, item_embedding

        else:

            gcl_loss = self.cal_gcl_loss(
                adj_norm=adj_norm, users=users, pos_items=pos_items, neg_items=neg_items
            )

            user_embedding = self.E_u
            item_embedding = self.E_i

            user_embedding = (
                user_embedding
                + args.model_cat_rate * F.normalize(image_user_embeds, p=2, dim=1)
                + args.model_cat_rate * F.normalize(text_user_embeds, p=2, dim=1)
            )
            item_embedding = (
                item_embedding
                + args.model_cat_rate * F.normalize(image_item_embeds, p=2, dim=1)
                + args.model_cat_rate * F.normalize(text_item_embeds, p=2, dim=1)
            )

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

        user_embedding = (
            self.user_id_embedding_pre.weight + self.user_id_embedding.weight
        )
        item_embedding = (
            self.item_id_embedding_pre.weight + self.item_id_embedding.weight
        )

        gnn_layer = self.gnn_layer
        self.E_u_list = [None] * (gnn_layer + 1)
        self.E_i_list = [None] * (gnn_layer + 1)
        self.Z_u_list = [None] * (gnn_layer + 1)
        self.Z_i_list = [None] * (gnn_layer + 1)
        self.G_u_list = [None] * (gnn_layer + 1)
        self.G_i_list = [None] * (gnn_layer + 1)

        uids = torch.tensor(users).long().cuda()
        iids = torch.tensor(pos_items + neg_items).long().cuda()

        self.E_u_list[0] = self.G_u_list[0] = user_embedding
        self.E_i_list[0] = self.G_i_list[0] = item_embedding

        for layer in range(1, self.gnn_layer + 1):

            self.Z_u_list[layer] = torch.sparse.mm(adj_norm, self.E_i_list[layer - 1])
            self.Z_i_list[layer] = torch.sparse.mm(adj_norm.T, self.E_u_list[layer - 1])

            vt_ei = self.vt @ self.E_i_list[layer - 1]
            self.G_u_list[layer] = self.u_mul_s @ vt_ei
            ut_eu = self.ut @ self.E_u_list[layer - 1]
            self.G_i_list[layer] = self.v_mul_s @ ut_eu

            self.E_u_list[layer] = self.Z_u_list[layer]
            self.E_i_list[layer] = self.Z_i_list[layer]

        self.G_u = torch.stack(self.G_u_list, dim=1).mean(dim=1, keepdim=False)
        self.G_i = torch.stack(self.G_i_list, dim=1).mean(dim=1, keepdim=False)

        self.E_u = torch.stack(self.E_u_list, dim=1).mean(dim=1, keepdim=False)
        self.E_i = torch.stack(self.E_i_list, dim=1).mean(dim=1, keepdim=False)

        neg_score = torch.logsumexp((self.G_u[uids] @ self.E_u.T) / 0.2, dim=1).mean()
        neg_score += torch.logsumexp((self.G_i[iids] @ self.E_i.T) / 0.2, dim=1).mean()
        pos_score = (
            torch.clamp((self.G_u[uids] * self.E_u[uids]).sum(dim=1) / 0.2, -5.0, 5.0)
        ).mean() + (
            torch.clamp((self.G_i[iids] * self.E_i[iids]).sum(dim=1) / 0.2, -5.0, 5.0)
        ).mean()
        gcl_loss = -pos_score + neg_score
        gcl_loss *= len(self.G_u_list) + 1
        return gcl_loss


class Student_MLPGCL(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim,
        gnn_layer,
        ut,
        vt,
        u_mul_s,
        v_mul_s,
        image_feats=None,
        text_feats=None,
    ):
        super().__init__()

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

        self.act_layers = nn.ModuleList(
            [nn.LeakyReLU(negative_slope=0.5) for _ in range(self.gnn_layer)]
        )
        self.user_trans_layers = nn.ModuleList(
            [
                nn.Linear(self.embedding_dim, self.embedding_dim)
                for _ in range(self.gnn_layer)
            ]
        )
        self.item_trans_layers = nn.ModuleList(
            [
                nn.Linear(self.embedding_dim, self.embedding_dim)
                for _ in range(self.gnn_layer)
            ]
        )

    def get_embedding(self):
        return self.user_id_embedding, self.item_id_embedding

    def layer_forward(self, user_id_embedding, item_id_embedding, layer):
        user_out = (
            self.act_layers[layer](self.user_trans_layers[layer](user_id_embedding))
            + user_id_embedding
        )
        item_out = (
            self.act_layers[layer](self.item_trans_layers[layer](item_id_embedding))
            + item_id_embedding
        )
        return user_out, item_out

    def multi_layer_forward(self, user_id_embedding, item_id_embedding):
        for layer in range(self.gnn_layer):
            user_id_embedding, item_id_embedding = self.layer_forward(
                user_id_embedding, item_id_embedding, layer
            )
        return user_id_embedding, item_id_embedding

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

        self.image_user_embeds = image_user_embeds
        self.text_user_embeds = text_user_embeds
        self.image_item_embeds = image_item_embeds
        self.text_item_embeds = text_item_embeds

        user_embedding = (
            self.user_id_embedding_pre.weight
            + self.user_id_embedding.weight
            + args.model_cat_rate * F.normalize(self.image_user_embeds, p=2, dim=1)
            + args.model_cat_rate * F.normalize(self.text_user_embeds, p=2, dim=1)
        )
        item_embedding = (
            self.item_id_embedding_pre.weight
            + self.item_id_embedding.weight
            + args.model_cat_rate * F.normalize(self.image_item_embeds, p=2, dim=1)
            + args.model_cat_rate * F.normalize(self.text_item_embeds, p=2, dim=1)
        )

        if is_test:
            user_embedding, item_embedding = self.multi_layer_forward(
                user_embedding, item_embedding
            )
            return user_embedding, item_embedding

        else:

            gcl_loss = self.cal_gcl_loss(
                adj_norm=adj_norm, users=users, pos_items=pos_items, neg_items=neg_items
            )

            user_embedding, item_embedding = self.multi_layer_forward(
                user_embedding, item_embedding
            )

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

        user_embedding = (
            self.user_id_embedding_pre.weight + self.user_id_embedding.weight
        )
        item_embedding = (
            self.item_id_embedding_pre.weight + self.item_id_embedding.weight
        )

        gnn_layer = self.gnn_layer + 1
        self.E_u_list = [None] * (gnn_layer + 1)
        self.E_i_list = [None] * (gnn_layer + 1)
        self.Z_u_list = [None] * (gnn_layer + 1)
        self.Z_i_list = [None] * (gnn_layer + 1)
        self.G_u_list = [None] * (gnn_layer + 1)
        self.G_i_list = [None] * (gnn_layer + 1)

        uids = torch.tensor(users).long().cuda()
        iids = torch.tensor(pos_items + neg_items).long().cuda()

        self.E_u_list[0] = self.G_u_list[0] = user_embedding
        self.E_i_list[0] = self.G_i_list[0] = item_embedding

        user_embedding, item_embedding = self.layer_forward(
            user_embedding, item_embedding, 0
        )
        user_embedding, item_embedding = self.layer_forward(
            user_embedding, item_embedding, 1
        )

        self.E_u_list[1] = torch.sparse.mm(adj_norm, self.E_i_list[0])
        self.E_i_list[1] = torch.sparse.mm(adj_norm.T, self.E_u_list[0])

        vt_ei = self.vt @ self.E_i_list[0]
        self.G_u_list[1] = self.u_mul_s @ vt_ei
        ut_eu = self.ut @ self.E_u_list[0]
        self.G_i_list[1] = self.v_mul_s @ ut_eu

        self.E_u_list[2], self.E_i_list[2] = self.layer_forward(
            self.E_u_list[1], self.E_i_list[1], 0
        )
        self.G_u_list[2], self.G_i_list[2] = self.layer_forward(
            self.G_u_list[1], self.G_i_list[1], 0
        )

        self.E_u_list[3], self.E_i_list[3] = self.layer_forward(
            self.E_u_list[2], self.E_i_list[2], 1
        )
        self.G_u_list[3], self.G_i_list[3] = self.layer_forward(
            self.G_u_list[2], self.G_i_list[2], 1
        )

        self.G_u = torch.stack(self.G_u_list[1:4], dim=1).mean(dim=1, keepdim=False)
        self.G_i = torch.stack(self.G_i_list[1:4], dim=1).mean(dim=1, keepdim=False)

        self.E_u = torch.stack(self.E_u_list[1:4], dim=1).mean(dim=1, keepdim=False)
        self.E_i = torch.stack(self.E_i_list[1:4], dim=1).mean(dim=1, keepdim=False)

        neg_score = torch.logsumexp((self.G_u[uids] @ self.E_u.T) / 0.2, dim=1).mean()
        neg_score += torch.logsumexp((self.G_i[iids] @ self.E_i.T) / 0.2, dim=1).mean()
        pos_score = (
            torch.clamp((self.G_u[uids] * self.E_u[uids]).sum(dim=1) / 0.2, -5.0, 5.0)
        ).mean() + (
            torch.clamp((self.G_i[iids] * self.E_i[iids]).sum(dim=1) / 0.2, -5.0, 5.0)
        ).mean()
        gcl_loss = -pos_score + neg_score

        return gcl_loss


class Student_LightGCN(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim,
        gnn_layer,
        image_feats=None,
        text_feats=None,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_ui_layers = gnn_layer

        self.user_id_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

    def init_user_item_embed(self, pre_u_embed, pre_i_embed):
        self.user_id_embedding = nn.Embedding.from_pretrained(pre_u_embed, freeze=False)
        self.item_id_embedding = nn.Embedding.from_pretrained(pre_i_embed, freeze=False)

        self.user_id_embedding_pre = nn.Embedding.from_pretrained(
            pre_u_embed, freeze=False
        )
        self.item_id_embedding_pre = nn.Embedding.from_pretrained(
            pre_i_embed, freeze=False
        )

    def get_embedding(self):
        return self.user_id_embedding, self.item_id_embedding

    def forward(self, adj):

        ego_embeddings = torch.cat(
            (
                self.user_id_embedding.weight + self.user_id_embedding_pre.weight,
                self.item_id_embedding.weight + self.item_id_embedding_pre.weight,
            ),
            dim=0,
        )

        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items], dim=0
        )

        return u_g_embeddings, i_g_embeddings


class Student_GCN(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim,
        gnn_layer=2,
        drop_out=0.0,
        image_feats=None,
        text_feats=None,
    ):
        super(Student_GCN, self).__init__()
        self.embedding_dim = embedding_dim

        self.trans_user = nn.Linear(args.embed_size, args.embed_size).cuda()
        self.trans_item = nn.Linear(args.embed_size, args.embed_size).cuda()

    def forward(self, user_x, item_x, ui_graph, iu_graph):

        return self.trans_user(user_x), self.trans_item(item_x)

    def l2_loss(self):
        layer = self.layers.children()
        layer = next(iter(layer))
        loss = None

        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss


class GraphConvolution(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        dropout=0.0,
        is_sparse_inputs=False,
        bias=False,
        activation=F.relu,
        featureless=False,
    ):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless

        self.user_weight = nn.Parameter(torch.empty(input_dim, output_dim))
        self.item_weight = nn.Parameter(torch.empty(input_dim, output_dim))
        nn.init.xavier_uniform_(self.user_weight)
        nn.init.xavier_uniform_(self.item_weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, user_x, item_x, ui_graph, iu_graph):

        user_x = F.dropout(user_x, self.dropout)
        item_x = F.dropout(item_x, self.dropout)

        if not self.featureless:
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(user_x, self.user_weight)
                xw = torch.sparse.mm(item_x, self.item_weight)
            else:
                xw_user = torch.mm(user_x, self.user_weight)
                xw_item = torch.mm(item_x, self.item_weight)
        else:
            xw = self.weight
        out_user = torch.sparse.mm(ui_graph, xw_item)
        out_item = torch.sparse.mm(iu_graph, xw_user)

        if self.bias is not None:
            out += self.bias
        return self.activation(out_user), self.activation(out_item)


def sparse_dropout(x, rate, noise_shape):
    """
    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()
    i = x._indices()
    v = x._values()

    i = i[:, dropout_mask]
    v = v[dropout_mask]
    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
    out = out * (1.0 / (1 - rate))
    return out


def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)
    return res


class BLMLP(nn.Module):
    def __init__(self):
        super(BLMLP, self).__init__()
        self.W = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(args.student_embed_size, args.student_embed_size)
            )
        )
        self.act = nn.LeakyReLU(negative_slope=0.5)

    def forward(self, embeds):
        pass

    def featureExtract(self, embeds):
        return self.act(embeds @ self.W) + embeds

    def pairPred(self, embeds1, embeds2):
        return (self.featureExtract(embeds1) * self.featureExtract(embeds2)).sum(dim=-1)

    def crossPred(self, embeds1, embeds2):
        return self.featureExtract(embeds1) @ self.featureExtract(embeds2).T


class Student_MLP(nn.Module):
    def __init__(self):
        super(Student_MLP, self).__init__()

        self.user_trans = nn.Linear(args.embed_size, args.embed_size)
        self.item_trans = nn.Linear(args.embed_size, args.embed_size)
        nn.init.xavier_uniform_(self.user_trans.weight)
        nn.init.xavier_uniform_(self.item_trans.weight)

        self.MLP = BLMLP()

    def get_embedding(self):
        return self.user_id_embedding, self.item_id_embedding

    def forward(
        self,
        pre_user,
        pre_item,
    ):

        user_embed = self.user_trans(pre_user)
        item_embed = self.user_trans(pre_item)

        return user_embed, item_embed

    def init_user_item_embed(self, pre_u_embed, pre_i_embed):
        self.user_id_embedding = nn.Embedding.from_pretrained(pre_u_embed, freeze=False)
        self.item_id_embedding = nn.Embedding.from_pretrained(pre_i_embed, freeze=False)

    def pointPosPredictwEmbeds(self, uEmbeds, iEmbeds, ancs, poss):
        ancEmbeds = uEmbeds[ancs]
        posEmbeds = iEmbeds[poss]
        nume = self.MLP.pairPred(ancEmbeds, posEmbeds)
        return nume

    def pointNegPredictwEmbeds(self, embeds1, embeds2, nodes1, temp=1.0):
        pckEmbeds1 = embeds1[nodes1]
        preds = self.MLP.crossPred(pckEmbeds1, embeds2)
        return torch.exp(preds / temp).sum(-1)

    def pairPredictwEmbeds(self, uEmbeds, iEmbeds, ancs, poss, negs):
        ancEmbeds = uEmbeds[ancs]
        posEmbeds = iEmbeds[poss]
        negEmbeds = iEmbeds[negs]
        posPreds = self.MLP.pairPred(ancEmbeds, posEmbeds)
        negPreds = self.MLP.pairPred(ancEmbeds, negEmbeds)
        return posPreds - negPreds

    def predAll(self, pckUEmbeds, iEmbeds):
        return self.MLP.crossPred(pckUEmbeds, iEmbeds)

    def testPred(self, usr, trnMask):
        uEmbeds, iEmbeds = self.forward()
        allPreds = self.predAll(uEmbeds[usr], iEmbeds) * (1 - trnMask) - trnMask * 1e8
        return allPreds
