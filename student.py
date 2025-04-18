import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.functions import NormLayer


class Student_MLPGCL(nn.Module):
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

        self.act_layers = nn.ModuleList(
            [nn.LeakyReLU(negative_slope=0.5) for _ in range(self.gnn_layer)]
        )
        self.user_trans_layers = nn.ModuleList(
            [
                torch.nn.utils.spectral_norm(
                    nn.Linear(self.embedding_dim, self.embedding_dim)
                )
                for _ in range(self.gnn_layer)
            ]
        )
        self.item_trans_layers = nn.ModuleList(
            [
                torch.nn.utils.spectral_norm(
                    nn.Linear(self.embedding_dim, self.embedding_dim)
                )
                for _ in range(self.gnn_layer)
            ]
        )
        self.user_bn_layers = nn.ModuleList(
            [nn.LayerNorm(self.embedding_dim) for _ in range(self.gnn_layer)]
        )
        self.item_bn_layers = nn.ModuleList(
            [nn.LayerNorm(self.embedding_dim) for _ in range(self.gnn_layer)]
        )

        self.args = args
        self.norm_layer = NormLayer(args.s_norm_mode, args.s_norm_scale)

        self.final_user_embedding = nn.Embedding(n_users, embedding_dim)
        self.final_item_embedding = nn.Embedding(n_items, embedding_dim)
        self.final_user_embedding.weight = nn.Parameter(
            torch.zeros(n_users, embedding_dim)
        )
        self.final_item_embedding.weight = nn.Parameter(
            torch.zeros(n_items, embedding_dim)
        )

    def get_embedding(self):
        user_out, item_out = self.multi_layer_forward(
            self.user_id_embedding.weight, self.item_id_embedding.weight
        )
        return user_out, item_out

    def layer_forward(self, user_id_embedding, item_id_embedding, layer):
        user_out = (
            self.act_layers[layer](
                self.user_bn_layers[layer](
                    self.user_trans_layers[layer](user_id_embedding)
                )
            )
            + user_id_embedding
        )
        item_out = (
            self.act_layers[layer](
                self.item_bn_layers[layer](
                    self.item_trans_layers[layer](item_id_embedding)
                )
            )
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
        users=None,
        pos_items=None,
        neg_items=None,
        is_test=False,
    ):

        user_embedding = self.user_id_embedding.weight
        item_embedding = self.item_id_embedding.weight

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

        combined_embeddings = torch.cat((user_embedding, item_embedding), dim=0)

        norm_embeddings = self.norm_layer(combined_embeddings, self.adj)
        user_embedding, item_embedding = torch.split(
            norm_embeddings, [self.n_users, self.n_items], dim=0
        )

        if is_test:
            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)

            # start.record()
            user_embedding, item_embedding = self.multi_layer_forward(
                user_embedding, item_embedding
            )
            # end.record()
            # torch.cuda.synchronize()
            #
            # inference_time = start.elapsed_time(end) / 1000
            # print(inference_time)
            return user_embedding, item_embedding

        else:

            gcl_loss = self.cal_gcl_loss(
                adj_norm=adj_norm, users=users, pos_items=pos_items, neg_items=neg_items
            )

            gcl_loss += g_reg_loss * 1e-3

            user_embedding, item_embedding = self.multi_layer_forward(
                user_embedding, item_embedding
            )

            self.final_user_embedding.weight = nn.Parameter(user_embedding)
            self.final_item_embedding.weight = nn.Parameter(item_embedding)

            return user_embedding, item_embedding, gcl_loss

    def init_user_item_embed(
        self, user_pretrained_embedding, item_pretrained_embedding
    ):
        self.user_id_embedding = nn.Embedding.from_pretrained(
            self.user_id_embedding.weight + user_pretrained_embedding, freeze=False
        )
        self.item_id_embedding = nn.Embedding.from_pretrained(
            self.item_id_embedding.weight + item_pretrained_embedding, freeze=False
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

        gnn_layer = self.gnn_layer + 1
        self.E_u_list = [None] * (gnn_layer + 1)
        self.E_i_list = [None] * (gnn_layer + 1)
        self.TE_u_list = [None] * (gnn_layer + 1)
        self.TE_i_list = [None] * (gnn_layer + 1)
        self.G_u_list = [None] * (gnn_layer + 1)
        self.G_i_list = [None] * (gnn_layer + 1)

        uids = torch.tensor(users).long().cuda()
        iids = torch.tensor(pos_items + neg_items).long().cuda()

        self.E_u_list[0] = self.G_u_list[0] = user_embedding
        self.E_i_list[0] = self.G_i_list[0] = item_embedding
        self.TE_u_list[0] = self.user_id_embedding_pre.weight
        self.TE_i_list[0] = self.item_id_embedding_pre.weight

        self.E_u_list[1] = torch.sparse.mm(adj_norm, self.E_i_list[0])
        self.E_i_list[1] = torch.sparse.mm(adj_norm.T, self.E_u_list[0])

        vt_ei = self.vt @ self.E_i_list[0]
        self.G_u_list[1] = self.u_mul_s @ vt_ei
        ut_eu = self.ut @ self.E_u_list[0]
        self.G_i_list[1] = self.v_mul_s @ ut_eu

        self.TE_u_list[1] = torch.sparse.mm(adj_norm, self.TE_i_list[0])
        self.TE_i_list[1] = torch.sparse.mm(adj_norm.T, self.TE_u_list[0])

        for i in range(2, self.gnn_layer + 2):
            self.E_u_list[i], self.E_i_list[i] = self.layer_forward(
                self.E_u_list[i - 1], self.E_i_list[i - 1], i - 2
            )
            self.G_u_list[i], self.G_i_list[i] = self.layer_forward(
                self.G_u_list[i - 1], self.G_i_list[i - 1], i - 2
            )

        self.G_u = torch.stack(self.G_u_list[1 : self.gnn_layer + 1], dim=1).mean(
            dim=1, keepdim=False
        )
        self.G_i = torch.stack(self.G_i_list[1 : self.gnn_layer + 1], dim=1).mean(
            dim=1, keepdim=False
        )

        self.E_u = torch.stack(self.E_u_list[1 : self.gnn_layer + 1], dim=1).mean(
            dim=1, keepdim=False
        )
        self.E_i = torch.stack(self.E_i_list[1 : self.gnn_layer + 1], dim=1).mean(
            dim=1, keepdim=False
        )

        neg_score = torch.logsumexp(
            (self.G_u[uids] @ self.E_u[uids].T) / 0.2, dim=1
        ).mean()
        neg_score += torch.logsumexp(
            (self.G_i[iids] @ self.E_i[iids].T) / 0.2, dim=1
        ).mean()
        pos_score = (
            torch.clamp((self.G_u[uids] * self.E_u[uids]).sum(dim=1) / 0.2, -5.0, 5.0)
        ).mean() + (
            torch.clamp((self.G_i[iids] * self.E_i[iids]).sum(dim=1) / 0.2, -5.0, 5.0)
        ).mean()
        gcl_loss = -pos_score + neg_score

        if round(self.args.s_layer_gcl, 8) != 0:

            layer_neg_score = (
                torch.logsumexp(
                    (self.E_u_list[0][uids] @ self.TE_i_list[1][pos_items].T) / 0.2,
                    dim=1,
                ).mean()
                + torch.logsumexp(
                    (self.TE_u_list[1][uids] @ self.E_i_list[0][pos_items].T) / 0.2,
                    dim=1,
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
                (self.E_u_list[0][uids] * self.TE_i_list[1][pos_items]).sum(dim=1)
                / 0.2,
                -5.0,
                5.0,
            ).mean()
            +torch.clamp(
                (self.TE_u_list[1][uids] * self.E_i_list[0][pos_items]).sum(dim=1)
                / 0.2,
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
            layer_loss = -layer_pos_score + layer_neg_score

            gcl_loss += self.args.s_layer_gcl * layer_loss

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

    def RINCE(
        self,
        view1,
        view2,
        temperature: float = 0.15,
        b_cos: bool = True,
        q: float = 0.5,
        lam: float = 0.025,
    ):
        """
        Args:
            view1: (torch.Tensor - N x D)
            view2: (torch.Tensor - N x D)
            temperature: float
            b_cos: bool
            q: float
            lam: float

        Return: Average RINCE Loss
        """
        if b_cos:
            view1, view2 = self.normalize(view1, dim=1), self.normalize(view2, dim=1)

        pos_score = (view1 @ view2.T) / temperature
        pos_exp = torch.exp(pos_score)

        all_score = view1 @ view1.T / temperature
        all_exp = torch.exp(all_score)

        neg_exp = all_exp - torch.diag_embed(torch.diag(all_exp))

        pos_term = -torch.pow(pos_exp, q).mean()
        neg_term = torch.pow(
            lam * (pos_exp.sum(dim=1, keepdim=True) + neg_exp.sum(dim=1, keepdim=True)),
            q,
        ).mean()

        loss = pos_term / q + neg_term / q
        return loss
