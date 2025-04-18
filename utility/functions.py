import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from utility.parser import args


def csr_norm(csr_mat, mean_flag=False):
    rowsum = np.array(csr_mat.sum(1))
    rowsum = np.power(rowsum + 1e-8, -0.5).flatten()
    rowsum[np.isinf(rowsum)] = 0.0
    rowsum_diag = sp.diags(rowsum)

    colsum = np.array(csr_mat.sum(0))
    colsum = np.power(colsum + 1e-8, -0.5).flatten()
    colsum[np.isinf(colsum)] = 0.0
    colsum_diag = sp.diags(colsum)

    if mean_flag == False:
        return rowsum_diag * csr_mat * colsum_diag
    else:
        return rowsum_diag * csr_mat


def matrix_to_tensor(cur_matrix):
    if type(cur_matrix) != sp.coo_matrix:
        cur_matrix = cur_matrix.tocoo()
    indices = torch.from_numpy(
        np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64)
    )
    values = torch.from_numpy(cur_matrix.data)
    shape = torch.Size(cur_matrix.shape)

    return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()


def bpr_loss(users, pos_items, neg_items):
    pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
    neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

    regularizer = (
        1.0 / 2 * (users ** 2).sum()
        + 1.0 / 2 * (pos_items ** 2).sum()
        + 1.0 / 2 * (neg_items ** 2).sum()
    )
    regularizer = regularizer / args.batch_size

    maxi = F.logsigmoid(pos_scores - neg_scores)
    mf_loss = -torch.mean(maxi)

    emb_loss = eval(args.regs)[0] * regularizer
    return mf_loss, emb_loss


def bpr_loss_for_KD(users, pos_items, neg_items):
    pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
    neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

    regularizer = (
        1.0 / 2 * (users ** 2) + 1.0 / 2 * (pos_items ** 2) + 1.0 / 2 * (neg_items ** 2)
    )
    regularizer = regularizer / args.batch_size

    maxi = F.logsigmoid(pos_scores - neg_scores)

    mf_loss = -maxi

    emb_loss = eval(args.regs)[0] * regularizer
    reg_loss = 0.0
    return mf_loss, emb_loss, reg_loss


def feat_reg_loss_calculation(
    g_item_image, g_item_text, g_user_image, g_user_text, n_items
):
    feat_reg = (
        1.0 / 2 * (g_item_image ** 2).sum()
        + 1.0 / 2 * (g_item_text ** 2).sum()
        + 1.0 / 2 * (g_user_image ** 2).sum()
        + 1.0 / 2 * (g_user_text ** 2).sum()
    )
    feat_reg = feat_reg / n_items
    feat_emb_loss = args.feat_reg_decay * feat_reg
    return feat_emb_loss


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_random_state():
    np.random.seed(None)
    random.seed(None)
    torch.manual_seed(torch.initial_seed())
    torch.cuda.manual_seed_all(torch.initial_seed())


def svd_for_gcl(ui_graph_norm, svd_q=1):

    u, s, v = torch.svd_lowrank(ui_graph_norm, q=svd_q)
    u_mul_s = u @ torch.diag(s)
    v_mul_s = v @ torch.diag(s)

    return u.T, v.T, u_mul_s, v_mul_s


def mask_edges(adj_matrix, mask_rate=0.1):
    """屏蔽用户-物品邻接矩阵的一部分边"""
    num_edges = adj_matrix.nnz
    mask_num = int(mask_rate * num_edges)

    rows, cols = adj_matrix.nonzero()
    edges = list(zip(rows, cols))

    np.random.shuffle(edges)
    mask_edges = edges[:mask_num]

    masked_adj = adj_matrix.copy()
    for row, col in mask_edges:
        masked_adj[row, col] = 0

    return masked_adj


def sce_criterion(x, y, alpha=2, tip_rate=0.2):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    if tip_rate != 0:
        loss = loss_function(loss, tip_rate)
        return loss

    loss = loss.mean()

    return loss


def loss_function(pred, drop_rate):

    ind_sorted = np.argsort(pred.cpu().data).cuda()
    loss_sorted = pred[ind_sorted]

    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    loss_update = pred[ind_update]

    return loss_update.mean()


def calcRegLoss(self, params=None, model=None):
    ret = 0
    if params is not None:
        for W in params:
            ret += W.norm(2).square()
    if model is not None:
        for W in model.parameters():
            ret += W.norm(2).square()

    return ret


def distillation_sinkhorn(y, teacher_scores, diameter=1, blur=0.005, reach=1e3):
    from geomloss import SamplesLoss

    return SamplesLoss(
        "sinkhorn", p=2, blur=blur, diameter=diameter, reach=reach, backend="tensorized"
    )(y.unsqueeze(0), teacher_scores.unsqueeze(0))


def distillation(y, teacher_scores, reach=None):

    return nn.KLDivLoss()(y, teacher_scores)


def load_state_dict_partial(model, state_dict):

    model_dict = model.state_dict()

    state_dict = {
        k: v
        for k, v in state_dict.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }

    model_dict.update(state_dict)

    model.load_state_dict(model_dict)


class NormLayer(nn.Module):
    def __init__(self, norm_mode="None", norm_scale=None):
        """
        mode:
          'None' : No normalization
          'PN'   : PairNorm
          'PN-SI'  : Scale-Individually version of PairNorm
          'PN-SCS' : Scale-and-Center-Simultaneously version of PairNorm
          'LN': LayerNorm
          'CN': ContraNorm
        """
        super(NormLayer, self).__init__()
        self.mode = norm_mode
        self.scale = norm_scale

    def forward(self, x, adj=None, tau=1.0):
        if self.mode == "None":
            return x
        if self.mode == "LN":
            x = x - x.mean(dim=1, keepdim=True)
            x = nn.functional.normalize(x, dim=1)

        col_mean = x.mean(dim=0)
        if self.mode == "PN":
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean

        if self.mode == "CN":
            from time import time

            H = nn.functional.normalize(x, dim=1)
            sim = H.T @ H / tau

            sim = nn.functional.softmax(sim, dim=1)
            x_neg = H @ sim
            x = (1 + self.scale) * x - self.scale * x_neg

        if self.mode == "PN-SI":
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == "PN-SCS":
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


def gauss_potential(embs, t):

    embs = F.normalize(embs, p=2, dim=-1)

    dist_matrix = torch.cdist(embs, embs, p=2)

    gauss_pot = torch.exp(-t * dist_matrix ** 2)

    return torch.sum(gauss_pot) - torch.sum(torch.diag(gauss_pot))


def hom_pot(user_embs, item_embs, t):
    user_pot = gauss_potential(user_embs, t)
    item_pot = gauss_potential(item_embs, t)
    return user_pot + item_pot


def het_pot(user_embs, neg_item_embs, t):
    user_embs = F.normalize(user_embs, p=2, dim=-1)
    neg_item_embs = F.normalize(neg_item_embs, p=2, dim=-1)
    cos_sim = torch.mm(user_embs, neg_item_embs.T)
    cos_dist = 1 - cos_sim
    gauss_pot = torch.exp(-t * cos_dist ** 2)
    return torch.sum(gauss_pot)


def reg_loss(user_embs, pos_item_embs, neg_item_embs, t, b1, b2):
    hom = hom_pot(user_embs, pos_item_embs, t)
    het = het_pot(user_embs, neg_item_embs, t)
    return b1 * hom + b2 * het
