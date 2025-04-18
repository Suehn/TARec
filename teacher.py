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


from utility.functions import NormLayer, set_seed


class Teacher_Model_GCL(nn.Module):
    def __init__(
        self,
        adj,
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
        args,
    ):
        super(Teacher_Model_GCL, self).__init__()

        self.final_user_embedding = nn.Embedding(n_users, embedding_dim)
        self.final_item_embedding = nn.Embedding(n_items, embedding_dim)

        self.args = args

        self.adj = adj
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_ui_layers = self.gnn_layer = gnn_layer

        self.user_id_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        if args.t_init_method == "normal":
            nn.init.xavier_normal_(self.user_id_embedding.weight)
            nn.init.xavier_normal_(self.item_id_embedding.weight)
        else:
            nn.init.xavier_uniform_(self.user_id_embedding.weight)
            nn.init.xavier_uniform_(self.item_id_embedding.weight)

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

        self.is_softmax = args.is_softmax

        self.norm_layer = NormLayer(args.norm_mode, args.norm_scale)

        self.final_user_embedding.weight = nn.Parameter(
            self.user_id_embedding.weight.clone().detach()
        )
        self.final_item_embedding.weight = nn.Parameter(
            self.item_id_embedding.weight.clone().detach()
        )

    def mm(self, x, y):
        return torch.sparse.mm(x, y) if self.args.sparse else torch.mm(x, y)

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

    def RINCE(
        self,
        view1,
        view2,
        temperature: float = 0.15,
        b_cos: bool = True,
        q=0.5,
        lamda=0.01,
    ):
        """
        Args:
            view1: (torch.Tensor - N x D)
            view2: (torch.Tensor - N x D)
            temperature: float
            b_cos: bool

        Return: Average RINCE Loss
        """
        if b_cos:
            view1, view2 = self.normalize(view1, dim=1), self.normalize(view2, dim=1)

        score = (view1 @ view2.T) / temperature

        exp_score = torch.exp(score)

        exp_pos = torch.diag(exp_score)

        exp_neg = exp_score.sum(dim=1) - exp_pos

        loss = -exp_pos.pow(q) / q + (lamda * exp_neg).pow(q) / q

        return loss.mean()

    def WinfoNCE(self, view1, view2, temperature: float = 0.15):
        """
        Args:
            view1: (torch.Tensor - N x D)
            view2: (torch.Tensor - N x D)
            temperature: float

        Return: Average InfoNCE Loss
        """
        raise NotImplementedError("这个对比loss未实现")

    def get_embeddings(self):
        return self.user_id_embedding.weight, self.item_id_embedding.weight

    def get_mm_feats(self):
        return self.image_feats, self.text_feats

    def get_freeze_trans(self):
        return self.image_trans, self.text_trans

    def forward(
        self,
        ui_graph,
        iu_graph,
        prompt_module=None,
        users=None,
        pos_items=None,
        neg_items=None,
    ):
        args = self.args

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

        if users is not None:

            from utility.functions import reg_loss

            u_g_embeddings = self.user_id_embedding.weight
            i_g_embeddings = self.item_id_embedding.weight
            g_reg_loss = reg_loss(
                u_g_embeddings[users],
                i_g_embeddings[pos_items],
                i_g_embeddings[neg_items],
                t=2,
                b1=0.02,
                b2=0.00,
            )

        u_g_embeddings = (
            self.user_id_embedding.weight
            + args.soft_token_rate * F.normalize(prompt_user, p=2, dim=1)
        )
        i_g_embeddings = (
            self.item_id_embedding.weight
            + args.soft_token_rate * F.normalize(prompt_item, p=2, dim=1)
        )

        combined_embeddings = torch.cat((u_g_embeddings, i_g_embeddings), dim=0)

        norm_embeddings = self.norm_layer(combined_embeddings, self.adj)
        u_g_embeddings, i_g_embeddings = torch.split(
            norm_embeddings, [self.n_users, self.n_items], dim=0
        )

        if users is not None and (round(args.t_cl_loss_rate, 9) != 0.0):
            gcl_loss = self.cal_gcl_loss(
                adj_norm=ui_graph,
                u_g_embeddings=u_g_embeddings,
                i_g_embeddings=i_g_embeddings,
                users=users,
                pos_items=pos_items,
                neg_items=neg_items,
            )

            gcl_loss += g_reg_loss * 1e-3
        else:
            gcl_loss = 0.0

        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]

        for i in range(self.gnn_layer):
            if i == (self.gnn_layer - 1) and self.is_softmax:
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

        self.final_user_embedding = nn.Embedding.from_pretrained(u_g_embeddings)
        self.final_item_embedding = nn.Embedding.from_pretrained(i_g_embeddings)

        return (
            u_g_embeddings,
            i_g_embeddings,
            image_item_feats,
            text_item_feats,
            image_user_feats,
            text_user_feats,
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

        self.X_u_list = [None] * (gnn_layer + 1)
        self.X_i_list = [None] * (gnn_layer + 1)

        self.XG_u_list = [None] * (gnn_layer + 1)
        self.XG_i_list = [None] * (gnn_layer + 1)

        uids = torch.tensor(users).long().cuda()

        iids = torch.tensor(pos_items).long().cuda()

        self.E_u_list[0] = self.G_u_list[0] = self.XG_u_list[0] = user_embedding
        self.E_i_list[0] = self.G_i_list[0] = self.XG_i_list[0] = item_embedding

        for layer in range(1, self.gnn_layer + 1):

            self.Z_u_list[layer] = torch.sparse.mm(adj_norm, self.E_i_list[layer - 1])
            self.Z_i_list[layer] = torch.sparse.mm(adj_norm.T, self.E_u_list[layer - 1])

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

            if layer == self.gnn_layer and self.args.is_gcl_softmax:
                self.G_u_list[layer] = self.softmax(self.G_u_list[layer])
                self.G_i_list[layer] = self.softmax(self.G_i_list[layer])
                self.XG_u_list[layer] = self.softmax(self.XG_u_list[layer])
                self.XG_i_list[layer] = self.softmax(self.XG_i_list[layer])
                self.Z_u_list[layer] = self.softmax(self.Z_u_list[layer])
                self.Z_i_list[layer] = self.softmax(self.Z_i_list[layer])

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

        gcl_loss_user = self.InfoNCE(self.G_u[uids], self.E_u[uids], 0.2)
        gcl_loss_item = self.InfoNCE(self.G_i[iids], self.E_i[iids], 0.2)
        svd_gcl = gcl_loss_user + gcl_loss_item

        xgcl_user = self.InfoNCE(self.X_u[uids], self.E_u[uids])
        xgcl_item = self.InfoNCE(self.X_i[iids], self.E_i[iids])
        x_gcl = xgcl_user + xgcl_item

        layer_gcl_01 = self.InfoNCE(self.E_u_list[0][uids], self.E_i_list[1][pos_items])
        layer_gcl_01 += self.InfoNCE(
            self.E_u_list[1][uids], self.E_i_list[0][pos_items]
        )
        layer_gcl_01 += self.InfoNCE(self.E_u_list[0][uids], self.E_u_list[1][uids])
        layer_gcl_01 += self.InfoNCE(self.E_i_list[0][iids], self.E_i_list[1][iids])
        layer_gcl = layer_gcl_01

        if round(self.args.svd_layer_gcl, 8) != 0.0:
            svd_layer_gcl_01 = self.InfoNCE(
                self.G_u_list[0][uids], self.G_i_list[1][pos_items]
            )
            svd_layer_gcl_01 += self.InfoNCE(
                self.G_u_list[1][uids], self.G_i_list[0][pos_items]
            )
            svd_layer_gcl_01 += self.InfoNCE(
                self.G_u_list[0][uids], self.G_u_list[1][uids]
            )
            svd_layer_gcl_01 += self.InfoNCE(
                self.G_i_list[0][iids], self.G_i_list[1][iids]
            )

            svd_layer_gcl = svd_layer_gcl_01
        else:
            svd_layer_gcl = 0.0

        if round(self.args.x_layer_gcl, 8) != 0.0:

            x_layer_gcl_01 = self.InfoNCE(
                self.E_u_list[0][uids], self.X_i_list[1][pos_items]
            )
            x_layer_gcl_01 += self.InfoNCE(
                self.X_u_list[1][uids], self.E_i_list[0][pos_items]
            )
            x_layer_gcl_01 += self.InfoNCE(
                self.E_u_list[0][uids], self.X_u_list[1][uids]
            )
            x_layer_gcl_01 += self.InfoNCE(
                self.E_i_list[0][iids], self.X_i_list[1][iids]
            )
            x_layer_gcl = x_layer_gcl_01
        else:
            x_layer_gcl = 0.0

        gcl_loss = self.args.svd_gcl_rate * svd_gcl
        gcl_loss += self.args.x_gcl_rate * x_gcl
        gcl_loss += self.args.layer_gcl * layer_gcl
        gcl_loss += self.args.svd_layer_gcl * svd_layer_gcl
        gcl_loss += self.args.x_layer_gcl * x_layer_gcl

        ssm_loss = self.InfoNCE(self.E_u_list[0][uids], self.E_i_list[0][pos_items])

        ssm_loss *= self.args.ssm_rate

        coeff_sum = (
            self.args.svd_gcl_rate
            + self.args.x_gcl_rate
            + self.args.layer_gcl
            + self.args.svd_layer_gcl
            + self.args.x_layer_gcl
        )

        gcl_loss = self.args.svd_gcl_rate * svd_gcl
        gcl_loss += self.args.x_gcl_rate * x_gcl
        gcl_loss += self.args.layer_gcl * layer_gcl
        gcl_loss += self.args.svd_layer_gcl * svd_layer_gcl
        gcl_loss += self.args.x_layer_gcl * x_layer_gcl

        gcl_loss = gcl_loss * 5 / coeff_sum + ssm_loss

        return gcl_loss


class PromptLearner(nn.Module):
    def __init__(self, image_feats=None, text_feats=None, ui_graph=None, args=None):
        super().__init__()
        self.ui_graph = ui_graph
        self.args = args

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
                # print("already load hard token", time() - t1)
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
        # elif args.hard_token_type=='tsne':
        #     hard_token_image = TSNE(n_components=args.embed_size, n_iter=300).fit_transform(image_feats)
        #     hard_token_text = TSNE(n_components=args.embed_size, n_iter=300).fit_transform(text_feats)
        # elif args.hard_token_type=='lda':
        #     hard_token_image = LinearDiscriminantAnalysis(n_components=args.embed_size).fit_transform(image_feats)
        #     hard_token_text = LinearDiscriminantAnalysis(n_components=args.embed_size).fit_transform(text_feats)

        # self.item_hard_token = nn.Embedding.from_pretrained(torch.mean((torch.stack((torch.tensor(hard_token_image).float(), torch.tensor(hard_token_text).float()))), dim=0), freeze=False).cuda().weight
        # self.user_hard_token = nn.Embedding.from_pretrained(torch.mm(ui_graph, self.item_hard_token), freeze=False).cuda().weight

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
        # nn.init.xavier_uniform_(self.gnn_trans_user.weight)
        # nn.init.xavier_uniform_(self.gnn_trans_item.weight)
        # self.gnn_trans_user = self.gnn_trans_user.cuda()
        # self.gnn_trans_item = self.gnn_trans_item.cuda()
        # self.item_hard_token = torch.mean((torch.stack((torch.tensor(hard_token_image).float(), torch.tensor(hard_token_text).float()))), dim=0).cuda()

    def forward(self):

        args = self.args

        return F.dropout(
            self.trans_user(self.user_hard_token), args.prompt_dropout
        ), F.dropout(self.trans_item(self.item_hard_token), args.prompt_dropout)
