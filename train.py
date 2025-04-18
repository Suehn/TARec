import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import scipy.sparse as sp


from tqdm import tqdm
from time import time
import random
import setproctitle
from pprint import pprint
import pickle
import math
import os


import sys
from utility.functions import *


from teacher import Teacher_Model_GCL, PromptLearner


from teacher_assistant import Student_LightGCL
from student import Student_MLPGCL


from utility.parser import init_args, save_parms, format_args


args = init_args()
setproctitle.setproctitle(args.name)


from utility.functions import (
    csr_norm,
    matrix_to_tensor,
    bpr_loss,
    bpr_loss_for_KD,
    feat_reg_loss_calculation,
    svd_for_gcl,
    sce_criterion,
    calcRegLoss,
    distillation_sinkhorn,
    distillation,
    load_state_dict_partial,
)


from utility.logging import Logger
from datetime import datetime


from utility.batch_test import data_generator, test_torch


torch.set_float32_matmul_precision("high")


class Trainer(object):
    def __init__(self, args):
        super(Teacher_Model_GCL).__init__()

        self.args = args
        self.logger = self.init_logger()

        self.lr = args.lr * 2
        self.student_lr = args.student_lr
        self.batch_size = args.batch_size
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.epoch = args.epoch

        self.n_layers = args.n_layers
        self.embed_dim = args.embed_size

        self.ta_n_layers = args.ta_n_layers

        self.image_feats = np.load(
            os.path.join(args.data_path, args.dataset, "image_feat.npy")
        )
        self.text_feats = np.load(
            os.path.join(args.data_path, args.dataset, "text_feat.npy")
        )
        self.image_feat_dim = self.image_feats.shape[-1]
        self.text_feat_dim = self.text_feats.shape[-1]

        self.ui_graph = self.ui_graph_raw = pickle.load(
            open(os.path.join(args.data_path, args.dataset, "train_mat"), "rb")
        )

        if args.edge_mask != 0:
            self.ui_graph = mask_edges(self.ui_graph, args.edge_mask_rate)
            print(f"\n\n edge_mask_rate: {args.edge_mask_rate} \n\n")

        self.iu_graph = self.ui_graph.T
        self.ui_graph = csr_norm(self.ui_graph, mean_flag=True)
        self.iu_graph = csr_norm(self.iu_graph, mean_flag=True)
        self.n_users = self.ui_graph.shape[0]
        self.n_items = self.ui_graph.shape[1]
        self.adj = sp.vstack(
            [
                sp.hstack([self.ui_graph, sp.csr_matrix((self.n_users, self.n_users))]),
                sp.hstack([sp.csr_matrix((self.n_items, self.n_items)), self.iu_graph]),
            ]
        )

        self.ui_graph = matrix_to_tensor(self.ui_graph)
        self.iu_graph = matrix_to_tensor(self.iu_graph)
        self.adj = matrix_to_tensor(self.adj)

        self.q = args.q
        self.ut, self.vt, self.u_mul_s, self.v_mul_s = svd_for_gcl(
            self.ui_graph, self.q
        )

        self.teacher_model_type = args.teacher_model_type
        self.teacher_model = torch.compile(
            self.init_teacher_model(self.teacher_model_type),
            mode="max-autotune",
        )

        self.prompt_module = PromptLearner(
            self.image_feats, self.text_feats, self.ui_graph, self.args
        )
        self.prompt_module_ta = PromptLearner(
            self.image_feats, self.text_feats, self.ui_graph, self.args
        )

        self.teacher_assistant_model_type = args.teacher_assistant_model_type
        self.teacher_assistant_model = torch.compile(
            self.init_teacher_assist_model(self.teacher_assistant_model_type),
            mode="max-autotune",
        )
        self.teacher_model_dict_name = args.teacher_model_dict_name

        self.student_model_type = args.student_model_type
        self.mlp_n_layers = args.mlp_n_layers
        self.student_embed_dim = args.student_embed_size
        self.teacher_assistant_model_dict_name = args.teacher_assistant_model_dict_name
        self.student_model_dict_name = args.student_model_dict_name
        self.student_model = self.init_student_model(self.student_model_type)

        self.opt_T = optim.AdamW(
            [
                {"params": self.teacher_model.parameters()},
                {"params": self.prompt_module.parameters()},
            ],
            lr=self.lr,
            weight_decay=args.t_weight_decay,
        )
        self.data_generator = data_generator

    def init_logger(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"test_ipynb_{timestamp}_{args.dataset}_{args.name}.txt"
        logger = Logger(log_file_name, False, folder=args.dataset)

        logger.logging(f"PID: {os.getpid()}\n")
        logger.logging(f"args: {format_args(args)}")
        return logger

    def init_teacher_model(self, teacher_model_type):
        if teacher_model_type == "gcl":
            teacher_model = Teacher_Model_GCL(
                self.adj,
                self.n_users,
                self.n_items,
                self.embed_dim,
                self.n_layers,
                self.image_feats,
                self.text_feats,
                self.ut,
                self.vt,
                self.u_mul_s,
                self.v_mul_s,
                self.args,
            )
        elif teacher_model_type == "xgcl":
            teacher_model = Teacher_Model_XGCL(
                self.n_users,
                self.n_items,
                self.embed_dim,
                self.n_layers,
                self.image_feats,
                self.text_feats,
                self.ut,
                self.vt,
                self.u_mul_s,
                self.v_mul_s,
                self.args,
            )

        elif teacher_model_type == "gcn":
            teacher_model = Teacher_Model_GCN(
                self.n_users,
                self.n_items,
                self.embed_dim,
                self.n_layers,
                self.image_feats,
                self.text_feats,
                self.ut,
                self.vt,
                self.u_mul_s,
                self.v_mul_s,
                self.args,
            )

        else:
            raise NotImplementedError("选择的教师模型未实现")

        teacher_model = teacher_model.cuda()
        return teacher_model

    def init_teacher_assist_model(self, teacher_assist_model_type):
        if teacher_assist_model_type == "lightgcl":
            teacher_assistant_model = Student_LightGCL(
                self.adj,
                self.n_users,
                self.n_items,
                self.embed_dim,
                self.ta_n_layers,
                self.ut,
                self.vt,
                self.u_mul_s,
                self.v_mul_s,
                self.args,
            )
        else:
            raise NotImplementedError("选择的助教模型未实现")

        teacher_assistant_model = teacher_assistant_model.cuda()
        return teacher_assistant_model

    def init_student_model(self, student_model_type):
        if student_model_type == "mlpgcl":
            student_model = Student_MLPGCL(
                self.adj,
                self.n_users,
                self.n_items,
                self.student_embed_dim,
                self.mlp_n_layers,
                self.ut,
                self.vt,
                self.u_mul_s,
                self.v_mul_s,
                self.args,
            )
        else:
            raise NotImplementedError("所选的student模型没实现")

        student_model = student_model.cuda()
        return student_model

    def train_teacher(self):

        print("\n")
        self.logger.logging("🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸")
        self.logger.logging("🎓📘 Start training teacher model... 🚀✨")
        self.logger.logging(f"Teacher model type: {self.teacher_model_type}")
        self.logger.logging("🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸🔸\n")

        stopping_step = 0
        t_best_recall_20 = 0
        total_batch = self.data_generator.n_train // self.batch_size + 1
        best_epoch = 0
        best_model_path = None

        if args.init_teacher:
            t_directory = os.path.join(
                os.getcwd(),
                "Model/teacher",
                args.dataset,
            )
            self.teacher_model.load_state_dict(
                torch.load(
                    os.path.join(
                        t_directory,
                        self.teacher_model_dict_name,
                    )
                    + ".pt"
                )
            )
            print(f"Load teacher model from {t_directory}")

        # Modification start: Initialize best_recall and best_ndcg lists
        best_recall = None
        best_ndcg = None
        # Modification end

        for epoch in range(self.epoch):

            total_loss = 0.0
            total_bpr_loss = 0.0
            total_prompt_bpr_loss = 0.0
            total_image_bpr_loss = 0.0
            total_text_bpr_loss = 0.0
            total_bpr_reg_loss = 0.0
            total_feat_reg_loss = 0.0
            total_gcl_loss = 0.0

            for batch in tqdm(range(total_batch)):

                self.teacher_model.train()
                users, pos_items, neg_items = self.data_generator.sample()

                if (
                    self.teacher_model_type == "gcl"
                    or self.teacher_model_type == "gcn"
                    or self.teacher_model_type == "xgcl"
                ):
                    (
                        t_u_id_embed,
                        t_i_id_embed,
                        t_i_image_embed,
                        t_i_text_embed,
                        t_u_image_embed,
                        t_u_text_embed,
                        prompt_user,
                        prompt_item,
                        gcl_loss,
                    ) = self.teacher_model(
                        self.ui_graph,
                        self.iu_graph,
                        self.prompt_module,
                        users,
                        pos_items,
                        neg_items,
                    )
                else:
                    raise NotImplementedError("选择的教师模型未实现")

                t_bpr_loss, t_bpr_reg_loss = bpr_loss(
                    t_u_id_embed[users],
                    t_i_id_embed[pos_items],
                    t_i_id_embed[neg_items],
                )

                t_bpr_loss *= self.args.t_bpr_loss_rate

                t_image_bpr_loss, t_image_bpr_reg_loss = bpr_loss(
                    t_u_image_embed[users],
                    t_i_image_embed[pos_items],
                    t_i_image_embed[neg_items],
                )

                t_text_bpr_loss, t_text_bpr_reg_loss = bpr_loss(
                    t_u_text_embed[users],
                    t_i_text_embed[pos_items],
                    t_i_text_embed[neg_items],
                )

                t_prompt_bpr_loss, t_prompt_bpr_reg_loss = bpr_loss(
                    prompt_user[users],
                    prompt_item[pos_items],
                    prompt_item[neg_items],
                )

                t_feat_reg_loss = feat_reg_loss_calculation(
                    t_i_image_embed,
                    t_i_text_embed,
                    t_u_image_embed,
                    t_u_text_embed,
                    self.n_items,
                )

                t_prompt_bpr_rate = args.t_prompt_rate1
                t_multimodal_bpr_rate = args.t_feat_mf_rate
                if gcl_loss is not None:
                    gcl_loss *= args.t_cl_loss_rate
                else:
                    gcl_loss = 0.0

                t_batch_loss = (
                    t_bpr_loss
                    + t_prompt_bpr_rate * t_prompt_bpr_loss
                    + t_multimodal_bpr_rate * (t_image_bpr_loss + t_text_bpr_loss)
                    + (t_bpr_reg_loss + t_feat_reg_loss)
                    + gcl_loss
                )

                total_loss += t_batch_loss.item()
                total_bpr_loss += t_bpr_loss.item()
                total_prompt_bpr_loss += t_prompt_bpr_loss.item()
                total_image_bpr_loss += t_image_bpr_loss.item()
                total_text_bpr_loss += t_text_bpr_loss.item()
                total_bpr_reg_loss += t_bpr_reg_loss.item()
                total_feat_reg_loss += t_feat_reg_loss.item()
                if gcl_loss is not None and gcl_loss != 0.0:
                    total_gcl_loss += gcl_loss.item()
                else:
                    total_gcl_loss = 0

                self.opt_T.zero_grad()
                t_batch_loss.backward(retain_graph=False)
                self.opt_T.step()
                del (
                    t_i_id_embed,
                    t_i_image_embed,
                    t_i_text_embed,
                    t_u_image_embed,
                    t_u_text_embed,
                )
                if math.isnan(total_loss) == True:
                    self.logger.logging("ERROR: loss is nan.")
                    sys.exit()

            users_to_test = list(self.data_generator.test_set.keys())

            t_result = self.test(users_to_test, is_val=False, is_teacher=True)

            # Modification start: Only consider elements from index 1 onwards in recall and ndcg lists
            if best_recall is None:
                best_recall = t_result["recall"].copy()
                best_ndcg = t_result["ndcg"].copy()
                best_epoch = epoch + 1
                stopping_step = 0
                improved = True
            else:
                improved_metrics = sum(
                    1
                    for i in range(1, len(t_result["recall"]))
                    if t_result["recall"][i] > best_recall[i]
                )
                improved_metrics += sum(
                    1
                    for i in range(1, len(t_result["ndcg"]))
                    if t_result["ndcg"][i] > best_ndcg[i]
                )

                if improved_metrics >= 3 and epoch >= 8:
                    best_recall = t_result["recall"].copy()
                    best_ndcg = t_result["ndcg"].copy()
                    best_epoch = epoch + 1
                    stopping_step = 0
                    improved = True
                elif epoch >= 8:
                    stopping_step += 1
                    improved = False
            # Modification end

            if improved:
                directory = os.path.join(
                    os.getcwd(),
                    "Model/teacher",
                    args.dataset,
                )

                if not os.path.exists(directory):
                    os.makedirs(directory)

                great_model_path = os.path.join(directory, "teacher_model_great.pt")
                great_prompt_path = os.path.join(directory, "teacher_prompt_great.pt")

                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)

                if epoch >= 8:
                    torch.save(self.teacher_model.state_dict(), great_model_path)
                    torch.save(self.prompt_module.state_dict(), great_prompt_path)
                    # Log best recall and ndcg values
                    best_recall_str = ", ".join([f"{rec:.5f}" for rec in best_recall])
                    best_ndcg_str = ", ".join([f"{ndcg:.5f}" for ndcg in best_ndcg])
                    self.logger.logging(
                        f"🎉Best recall: [{best_recall_str}], Best ndcg: [{best_ndcg_str}]. Model saved to teacher_model_great.pt"
                    )

            if stopping_step == args.early_stopping_patience:
                self.logger.logging(f"early stopping at epoch {epoch + 1}")
                break

            avg_loss = total_loss / total_batch
            avg_bpr_loss = total_bpr_loss / total_batch
            avg_prompt_bpr_loss = total_prompt_bpr_loss / total_batch
            avg_image_bpr_loss = total_image_bpr_loss / total_batch
            avg_text_bpr_loss = total_text_bpr_loss / total_batch
            avg_bpr_reg_loss = total_bpr_reg_loss / total_batch
            avg_feat_reg_loss = total_feat_reg_loss / total_batch
            avg_gcl_loss = total_gcl_loss / total_batch

            loss_string = (
                f"🎓📘Epoch {epoch+1}/{args.epoch} Early stopping {stopping_step} - "
                f"Recall {t_result['recall'][1]:.5f}, Ndcg: {t_result['ndcg'][1]:.4f} || "
                f"Avg Loss: {avg_loss:.4f} | BPR: {avg_bpr_loss:.4f}, "
                f"Prompt: {avg_prompt_bpr_loss:.4f}, Image: {avg_image_bpr_loss:.4f}, "
                f"Text: {avg_text_bpr_loss:.4f}, Reg: {avg_bpr_reg_loss:.4f}, Feat_Reg: {avg_feat_reg_loss:.4f}, "
                f"GCL: {avg_gcl_loss:.4f}"
            )
            self.logger.logging(loss_string)

        directory = os.path.join(
            os.getcwd(),
            "Model/teacher",
            args.dataset,
        )
        final_model_path = os.path.join(directory, "teacher_model_final.pt")
        torch.save(self.teacher_model.state_dict(), final_model_path)
        # Modification start: Adjust logging of best recall and ndcg values
        best_recall_str = ", ".join([f"{rec:.5f}" for rec in best_recall])
        best_ndcg_str = ", ".join([f"{ndcg:.5f}" for ndcg in best_ndcg])
        self.logger.logging(
            f"🏆🎉Final model saved to {final_model_path}, best epoch: {best_epoch}, best recall: [{best_recall_str}], best ndcg: [{best_ndcg_str}]"
        )
        # Modification end
        self.logger.logging("⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛")
        self.logger.logging("✅🎓📘 Finished training teacher model... 🏆🎉")
        self.logger.logging("⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛")

    def train_teacher_assistant(self):

        print("\n")
        self.logger.logging("🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶")
        self.logger.logging("🧑📘 Start training teacher_assistant model... 🚀✨")
        self.logger.logging(
            f"🎓Teacher:{self.teacher_model_type} || 🧑📘TA: {self.teacher_assistant_model_type}"
        )
        self.logger.logging("🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶\n")

        stopping_step = 0
        ta_best_recall_20 = 0
        total_batch = self.data_generator.n_train // self.batch_size + 1
        best_epoch = 0
        best_model_path = None

        t_directory = os.path.join(
            os.getcwd(),
            "Model/teacher",
            args.dataset,
        )
        self.teacher_model.load_state_dict(
            torch.load(
                os.path.join(
                    t_directory,
                    self.teacher_model_dict_name,
                )
                + ".pt"
            )
        )
        self.prompt_module_ta.load_state_dict(
            torch.load(
                os.path.join(
                    t_directory,
                    "teacher_prompt_great",
                )
                + ".pt"
            )
        )

        self.prompt_module.load_state_dict(
            torch.load(
                os.path.join(
                    t_directory,
                    "teacher_prompt_great",
                )
                + ".pt"
            )
        )

        self.teacher_model.eval()
        self.logger.logging(f"🎓load teacher model {self.teacher_model_dict_name}.pt")

        num_seeds = 10

        best_recall = -1
        best_t_u_embed = None
        best_t_i_embed = None

        for i in range(num_seeds):

            seed = np.random.randint(0, 2**16)

            set_seed(seed)
            with torch.no_grad():

                if i != 0:
                    torch.nn.init.xavier_uniform_(self.prompt_module.trans_item.weight)
                    torch.nn.init.xavier_uniform_(self.prompt_module.trans_user.weight)

                t_u_embed, t_i_embed, *rest = self.teacher_model(
                    self.ui_graph, self.iu_graph, self.prompt_module
                )

                users_to_test = list(self.data_generator.test_set.keys())
                t_result = self.test(users_to_test, is_val=False, is_teacher=True)

                recall_at_20 = t_result["recall"][1]
                # self.logger.logging(
                #     f"🎓Teacher (Seed {seed}): Recall@20: {recall_at_20:.5f}"
                # )

                if recall_at_20 > best_recall:
                    best_recall = recall_at_20
                    best_t_u_embed = t_u_embed
                    best_t_i_embed = t_i_embed

        t_u_embed = best_t_u_embed
        t_i_embed = best_t_i_embed

        self.teacher_assistant_model.init_user_item_embed(t_u_embed, t_i_embed)

        image_feats, text_feats = (
            torch.tensor(self.image_feats).float().cuda(),
            torch.tensor(self.text_feats).float().cuda(),
        )

        image_trans, text_trans = self.teacher_model.get_freeze_trans()

        new_image_trans = nn.Linear(image_trans.in_features, image_trans.out_features)
        new_image_trans.cuda().load_state_dict(image_trans.state_dict())

        new_text_trans = nn.Linear(text_trans.in_features, text_trans.out_features)
        new_text_trans.cuda().load_state_dict(text_trans.state_dict())

        item_hard_token = self.prompt_module_ta.item_hard_token
        user_hard_token = self.prompt_module_ta.user_hard_token
        old_trans_user, old_trans_item = (
            self.prompt_module_ta.trans_user,
            self.prompt_module_ta.trans_item,
        )
        trans_user = nn.Linear(self.args.embed_size, self.args.embed_size).cuda()
        trans_item = nn.Linear(self.args.embed_size, self.args.embed_size).cuda()
        trans_user.load_state_dict(old_trans_user.state_dict())
        trans_item.load_state_dict(old_trans_item.state_dict())

        self.opt_TA = optim.AdamW(
            [
                {"params": self.teacher_assistant_model.parameters()},
                {"params": self.prompt_module_ta.parameters()},
                {"params": new_image_trans.parameters()},
                {"params": new_text_trans.parameters()},
                {"params": trans_user.parameters()},
                {"params": trans_item.parameters()},
            ],
            lr=self.student_lr,
        )

        for epoch in range(self.epoch):

            total_loss = 0.0
            total_bpr_loss = 0.0
            total_pure_ranking_kd_loss = 0.0
            total_kd_loss_feat = 0.0
            total_gcl_loss = 0.0
            total_reg_loss = 0.0

            for batch in tqdm(range(total_batch)):

                prompt_user, prompt_item = trans_user(user_hard_token), trans_item(
                    item_hard_token
                )
                feat_prompt_item_image = torch.mm(
                    prompt_item, torch.mm(prompt_item.T, image_feats)
                )
                feat_prompt_item_text = torch.mm(
                    prompt_item, torch.mm(prompt_item.T, text_feats)
                )

                Drop_out = nn.Dropout(p=args.drop_rate)
                image_feat = Drop_out(
                    new_image_trans(
                        image_feats
                        + args.feat_soft_token_rate
                        * F.normalize(feat_prompt_item_image)
                    )
                )
                text_feat = Drop_out(
                    new_text_trans(
                        text_feats
                        + args.feat_soft_token_rate * F.normalize(feat_prompt_item_text)
                    )
                )

                for _ in range(self.args.n_layers):
                    image_user_feats = self.mm(self.ui_graph, image_feat)
                    image_item_feats = self.mm(self.iu_graph, image_user_feats)

                    text_user_feats = self.mm(self.ui_graph, text_feat)
                    text_item_feats = self.mm(self.iu_graph, text_user_feats)

                t_i_image_embed = image_item_feats
                t_i_text_embed = text_item_feats
                t_u_image_embed = image_user_feats
                t_u_text_embed = text_user_feats

                ta_gcl_loss = 0.0
                self.teacher_assistant_model.train()
                users, pos_items, neg_items = self.data_generator.sample()

                if self.teacher_assistant_model_type == "lightgcl":
                    ta_u_embed, ta_i_embed, ta_gcl_loss = self.teacher_assistant_model(
                        self.ui_graph,
                        t_i_image_embed,
                        t_i_text_embed,
                        t_u_image_embed,
                        t_u_text_embed,
                        users,
                        pos_items,
                        neg_items,
                    )
                else:
                    raise NotImplementedError("所选的助教模型未实现")

                ta_bpr_loss, ta_bpr_reg_loss = bpr_loss(
                    ta_u_embed[users], ta_i_embed[pos_items], ta_i_embed[neg_items]
                )

                ta_bpr_score_for_KD, *res = bpr_loss_for_KD(
                    ta_u_embed[users], ta_i_embed[pos_items], ta_i_embed[neg_items]
                )
                t_bpr_score_for_KD, *res = bpr_loss_for_KD(
                    t_u_embed[users], t_i_embed[pos_items], t_i_embed[neg_items]
                )

                if args.kd_loss_type == "sinkhorn":
                    distill = distillation_sinkhorn
                elif args.kd_loss_type == "kl":
                    distill = distillation
                else:
                    raise NotImplemented("所选的蒸馏函数未实现")

                if round(args.kd_loss_rate, 8) != 0.0:
                    pure_ranking_kd_loss = 0.0
                    if ta_best_recall_20 < t_result["recall"][1]:
                        pure_ranking_kd_loss += distill(
                            ta_bpr_score_for_KD, t_bpr_score_for_KD, reach=2.5e2
                        )
                    elif round(args.kd_t_decay_rate, 8) != 0.0:
                        pure_ranking_kd_loss += args.kd_t_decay_rate * distill(
                            ta_bpr_score_for_KD,
                            t_bpr_score_for_KD,
                            reach=2.5e2 * args.kd_t_decay_rate,
                        )
                    else:
                        pure_ranking_kd_loss = 0.0
                else:
                    pure_ranking_kd_loss = 0.0

                kd_loss_feat = (
                    sce_criterion(t_i_image_embed, ta_i_embed, alpha=2, tip_rate=0.2)
                    + sce_criterion(t_i_text_embed, ta_i_embed, alpha=2, tip_rate=0.2)
                    + sce_criterion(t_u_image_embed, ta_u_embed, alpha=2, tip_rate=0.2)
                    + sce_criterion(t_u_text_embed, ta_u_embed, alpha=2, tip_rate=0.2)
                )

                ta_reg_loss = calcRegLoss([ta_u_embed, ta_i_embed]) * args.emb_reg

                ta_batch_loss = (
                    ta_bpr_loss
                    + pure_ranking_kd_loss * args.kd_loss_rate
                    + kd_loss_feat * args.kd_loss_feat_rate
                    + ta_gcl_loss * args.cl_loss_rate
                    + ta_reg_loss
                    + ta_bpr_reg_loss
                )
                if math.isnan(ta_batch_loss) == True:
                    self.logger.logging("ERROR: loss is nan.")
                    sys.exit()

                total_loss += ta_batch_loss.item()
                total_bpr_loss += ta_bpr_loss.item()
                if pure_ranking_kd_loss != 0.0:
                    total_pure_ranking_kd_loss += pure_ranking_kd_loss.item()
                else:
                    total_pure_ranking_kd_loss += 0.0
                total_kd_loss_feat += kd_loss_feat.item()
                total_gcl_loss += ta_gcl_loss.item()
                total_reg_loss += ta_reg_loss

                self.opt_TA.zero_grad()
                ta_batch_loss.backward()
                self.opt_TA.step()

            users_to_test = list(data_generator.test_set.keys())
            ta_result = self.test(
                users_to_test, is_val=False, is_teacher=False, is_teacher_assistant=True
            )

            avg_loss = total_loss / total_batch
            avg_bpr_loss = total_bpr_loss / total_batch
            avg_pure_ranking_kd_loss = total_pure_ranking_kd_loss / total_batch
            avg_kd_loss_feat = total_kd_loss_feat / total_batch
            avg_gcl_loss = total_gcl_loss / total_batch
            avg_reg_loss = total_reg_loss / total_batch

            loss_string = (
                f"🧑📘Epoch {epoch+1}/{args.epoch} Early stopping {stopping_step} - "
                f"Recall {ta_result['recall'][1]:.5f}, Ndcg: {ta_result['ndcg'][1]:.4f} || "
                f"Avg Loss: {avg_loss:.4f} | BPR: {avg_bpr_loss:.4f}, "
                f"Pure Ranking KD: {avg_pure_ranking_kd_loss:.4f}, KD Feat: {avg_kd_loss_feat:.4f}, "
                f"GCL: {avg_gcl_loss:.4f}, Reg: {avg_reg_loss:.4f}"
            )
            self.logger.logging(loss_string)

            if ta_result["recall"][1] > ta_best_recall_20:
                ta_best_recall_20 = ta_result["recall"][1]
                best_epoch = epoch + 1
                stopping_step = 0

                directory = os.path.join(
                    os.getcwd(),
                    "Model/ta",
                    args.dataset,
                )

                new_best_model_path = os.path.join(
                    directory,
                    f"ta_model_great_layers{self.ta_n_layers}seed{args.seed}_{args.embed_size}_{ta_best_recall_20:.5f}_e{best_epoch}.pt",
                )

                if not os.path.exists(directory):
                    os.makedirs(directory)

                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)

                best_model_path = new_best_model_path
                # torch.save(self.teacher_assistant_model.state_dict(), best_model_path)
                self.logger.logging(
                    f"🎉Best recall@20: {ta_best_recall_20:.5f}.Model saved to ta_model_great.pt"
                )
                great_model_path = os.path.join(
                    os.getcwd(), "Model/ta", args.dataset, "ta_model_great.pt"
                )
                torch.save(self.teacher_assistant_model.state_dict(), great_model_path)
            else:
                stopping_step += 1

            if stopping_step == args.early_stopping_patience:
                self.logger.logging(f"early stopping at epoch {epoch + 1}")
                break

        final_model_path = os.path.join(
            os.getcwd(), "Model/ta", args.dataset, "ta_model_final.pt"
        )
        torch.save(self.teacher_assistant_model.state_dict(), final_model_path)
        self.logger.logging(
            f"Final ta model saved to {final_model_path}, best epoch: {best_epoch}, best recall@20: {ta_best_recall_20:.5f}"
        )

        self.logger.logging("⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛")
        self.logger.logging("✅🧑📘 Finished training ta model... 🏆🎉")
        self.logger.logging(
            f"🎓Teacher:{self.teacher_model_type} || 🧑📘ta: {self.teacher_assistant_model_type}"
        )
        self.logger.logging("⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛")

    def train_student(self):

        print("\n")
        self.logger.logging("🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶")
        self.logger.logging("🧑📘 Start training student model... 🚀✨")
        self.logger.logging(
            f"🎓Teacher:{self.teacher_assistant_model_type} || 🧑Student: {self.student_model_type}"
        )
        self.logger.logging("🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶🔶\n")

        stopping_step = 0
        s_best_recall_20 = 0
        total_batch = self.data_generator.n_train // self.batch_size + 1
        best_epoch = 0
        best_model_path = None

        self.teacher_model.load_state_dict(
            torch.load(
                os.path.join(
                    os.getcwd(),
                    "Model/teacher",
                    args.dataset,
                    self.teacher_model_dict_name,
                )
                + ".pt"
            )
        )
        self.teacher_model.eval()
        self.logger.logging(f"🎓load teacher model {self.teacher_model_dict_name}.pt")

        self.teacher_assistant_model.load_state_dict(
            torch.load(
                os.path.join(
                    os.getcwd(),
                    "Model/ta",
                    args.dataset,
                    self.teacher_assistant_model_dict_name,
                )
                + ".pt"
            )
        )
        self.teacher_assistant_model.eval()
        self.logger.logging(
            f"🎓load teacher assistant model {self.teacher_assistant_model_dict_name}.pt"
        )

        with torch.no_grad():
            (
                t_u_embed,
                t_i_embed,
                t_i_image_embed,
                t_i_text_embed,
                t_u_image_embed,
                t_u_text_embed,
                *rest,
            ) = self.teacher_model(self.ui_graph, self.iu_graph, self.prompt_module)

            users_to_test = list(self.data_generator.test_set.keys())

            t_result = self.test(users_to_test, is_val=False, is_teacher=True)
            t_recall = t_result["recall"][1]
            self.logger.logging(f"🎓Teacher: Recall@20: {t_result['recall'][1]:.5f}")

        with torch.no_grad():
            ta_u_embed, ta_i_embed, *res = self.teacher_assistant_model(
                self.ui_graph,
                t_i_image_embed,
                t_i_text_embed,
                t_u_image_embed,
                t_u_text_embed,
                users=None,
                pos_items=None,
                neg_items=None,
                is_test=True,
            )
            ta_result = self.test(
                users_to_test, is_val=False, is_teacher=False, is_teacher_assistant=True
            )
            ta_recall = ta_result["recall"][1]
            self.logger.logging(
                f"🎓Teacher Assistant : Recall@20: {ta_result['recall'][1]:.5f}"
            )

            self.student_model.init_user_item_embed(t_u_embed, t_i_embed)

            if self.student_model_dict_name != "":
                state_dict_path = (
                    os.path.join(
                        os.getcwd(),
                        "Model/student",
                        args.dataset,
                        self.student_model_dict_name,
                    )
                    + ".pt"
                )
                student_state_dict = torch.load(state_dict_path)

                self.logger.logging(
                    f"🧒load student model {self.student_model_dict_name}.pt"
                )

                self.student_model.init_user_item_embed(
                    t_u_embed + 0.4 * student_state_dict["user_id_embedding.weight"],
                    t_i_embed + 0.4 * student_state_dict["item_id_embedding.weight"],
                )
                self.student_model.load_state_dict(student_state_dict)

        self.opt_S = optim.AdamW(
            [
                {"params": self.student_model.parameters()},
                {"params": self.prompt_module.parameters()},
            ],
            lr=self.student_lr,
        )

        for epoch in range(args.epoch):
            start_time = time()
            total_loss = 0.0
            total_bpr_loss = 0.0
            total_pure_ranking_kd_loss = 0.0
            total_kd_loss_feat = 0.0
            total_gcl_loss = 0.0
            total_reg_loss = 0.0

            for batch in tqdm(range(total_batch)):
                s_gcl_loss = 0.0

                self.student_model.train()
                users, pos_items, neg_items = self.data_generator.sample()

                if self.student_model_type == "lightgcl":
                    s_u_embed, s_i_embed, s_gcl_loss = self.student_model(
                        self.ui_graph,
                        t_i_image_embed,
                        t_i_text_embed,
                        t_u_image_embed,
                        t_u_text_embed,
                        users,
                        pos_items,
                        neg_items,
                    )

                elif self.student_model_type == "mlpgcl":
                    s_u_embed, s_i_embed, s_gcl_loss = self.student_model(
                        self.ui_graph,
                        users,
                        pos_items,
                        neg_items,
                        is_test=False,
                    )

                elif self.student_model_type == "lightgcn":
                    raise NotImplementedError("未添加该模型")

                elif self.student_model_type == "mlp":
                    raise NotImplementedError("未添加该模型")

                s_bpr_loss, s_bpr_reg_loss = bpr_loss(
                    s_u_embed[users], s_i_embed[pos_items], s_i_embed[neg_items]
                )

                s_bpr_score_for_KD, *res = bpr_loss_for_KD(
                    s_u_embed[users], s_i_embed[pos_items], s_i_embed[neg_items]
                )

                if not args.is_teacher_kd:

                    ta_bpr_score_for_KD, *res = bpr_loss_for_KD(
                        ta_u_embed[users], ta_i_embed[pos_items], ta_i_embed[neg_items]
                    )
                else:

                    ta_bpr_score_for_KD, *res = bpr_loss_for_KD(
                        t_u_embed[users], t_i_embed[pos_items], t_i_embed[neg_items]
                    )

                if args.kd_loss_type == "sinkhorn":
                    distill = distillation_sinkhorn
                elif args.kd_loss_type == "kl":
                    distill = distillation
                else:
                    raise NotImplemented("所选的蒸馏函数未实现")

                if round(args.kd_loss_rate, 8) != 0.0:
                    pure_ranking_kd_loss = 0.0

                    if ta_recall >= s_best_recall_20:
                        pure_ranking_kd_loss += distill(
                            s_bpr_score_for_KD,
                            ta_bpr_score_for_KD,
                        )
                    elif round(args.kd_ta_decay_rate, 8) != 0.0:
                        pure_ranking_kd_loss += args.kd_ta_decay_rate * distill(
                            s_bpr_score_for_KD,
                            ta_bpr_score_for_KD,
                        )
                else:
                    pure_ranking_kd_loss = 0.0

                kd_loss_feat = (
                    sce_criterion(t_i_image_embed, s_i_embed, alpha=2, tip_rate=0.2)
                    + sce_criterion(t_i_text_embed, s_i_embed, alpha=2, tip_rate=0.2)
                    + sce_criterion(t_u_image_embed, s_u_embed, alpha=2, tip_rate=0.2)
                    + sce_criterion(t_u_text_embed, s_u_embed, alpha=2, tip_rate=0.2)
                )

                s_reg_loss = calcRegLoss([s_u_embed, s_i_embed]) * args.emb_reg

                student_batch_loss = (
                    s_bpr_loss
                    + pure_ranking_kd_loss * args.kd_loss_rate
                    + kd_loss_feat * args.kd_loss_feat_rate
                    + s_gcl_loss * args.cl_loss_rate
                    + s_reg_loss
                    + s_bpr_reg_loss
                )
                if math.isnan(student_batch_loss) == True:
                    self.logger.logging("ERROR: loss is nan.")
                    sys.exit()

                total_loss += student_batch_loss.item()
                total_bpr_loss += s_bpr_loss.item()
                if pure_ranking_kd_loss != 0.0:
                    total_pure_ranking_kd_loss += pure_ranking_kd_loss.item()
                else:
                    total_pure_ranking_kd_loss += 0.0
                total_kd_loss_feat += kd_loss_feat.item()
                total_gcl_loss += s_gcl_loss.item()
                total_reg_loss += s_reg_loss

                self.opt_S.zero_grad()
                student_batch_loss.backward()
                self.opt_S.step()

            users_to_test = list(data_generator.test_set.keys())
            s_result = self.test(users_to_test, is_val=False, is_teacher=False)

            avg_loss = total_loss / total_batch
            avg_bpr_loss = total_bpr_loss / total_batch
            avg_pure_ranking_kd_loss = total_pure_ranking_kd_loss / total_batch
            avg_kd_loss_feat = total_kd_loss_feat / total_batch
            avg_gcl_loss = total_gcl_loss / total_batch
            avg_reg_loss = total_reg_loss / total_batch

            loss_string = (
                f"🧑📘Epoch {epoch+1}/{args.epoch} Early stopping {stopping_step} - "
                f"Recall {s_result['recall'][1]:.5f}/{s_result['recall'][-1]:.5f}, Ndcg: {s_result['ndcg'][1]:.4f}/{s_result['ndcg'][-1]:.4f}  || "
                f"Avg Loss: {avg_loss:.4f} | BPR: {avg_bpr_loss:.4f}, "
                f"Pure Ranking KD: {avg_pure_ranking_kd_loss:.4f}, KD Feat: {avg_kd_loss_feat:.4f}, "
                f"GCL: {avg_gcl_loss:.4f}, Reg: {avg_reg_loss:.4f}"
            )
            self.logger.logging(loss_string)

            if s_result["recall"][1] > s_best_recall_20:
                s_best_recall_20 = s_result["recall"][1]
                best_epoch = epoch + 1
                stopping_step = 0

                s_directory = os.path.join(
                    os.getcwd(),
                    "Model/student",
                    args.dataset,
                )
                new_best_model_path = os.path.join(
                    s_directory,
                    f"student_model_great_layer{self.mlp_n_layers}_seed{args.seed}_{args.embed_size}_{s_best_recall_20:.5f}_e{best_epoch}.pt",
                )

                if not os.path.exists(s_directory):
                    os.makedirs(s_directory)

                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)

                best_model_path = new_best_model_path
                # torch.save(self.student_model.state_dict(), best_model_path)
                self.logger.logging(
                    f"🎉Best recall@20: {s_best_recall_20:.5f}.Model saved to student_model_great.pt"
                )
            else:
                stopping_step += 1

            if stopping_step == args.early_stopping_patience:
                self.logger.logging(f"early stopping at epoch {epoch + 1}")
                break

        final_model_path = os.path.join(
            os.getcwd(), "Model", args.dataset, "student_model_final.pt"
        )
        torch.save(self.student_model.state_dict(), final_model_path)
        self.logger.logging(f"Final model saved to {final_model_path}")

        self.logger.logging("⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛")
        self.logger.logging(
            f"✅🧑📘 Finished training student model... 🏆🎉 best_epoch:{best_epoch}, ta_best_recall_20{s_best_recall_20}"
        )
        self.logger.logging(
            f"🎓Teacher:{self.teacher_model_type} || 🧑Student: {self.student_model_type}"
        )
        self.logger.logging("⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛")

    def test(
        self, users_to_test, is_val=False, is_teacher=True, is_teacher_assistant=False
    ):

        self.teacher_model.eval()
        with torch.no_grad():
            if is_teacher:
                u_embed, i_embed, *rest = self.teacher_model(
                    self.ui_graph, self.iu_graph, self.prompt_module
                )

            elif is_teacher_assistant:
                (
                    _,
                    _,
                    t_i_image_embed,
                    t_i_text_embed,
                    t_u_image_embed,
                    t_u_text_embed,
                    *rest,
                ) = self.teacher_model(self.ui_graph, self.iu_graph, self.prompt_module)
                u_embed, i_embed, *res = self.teacher_assistant_model(
                    self.ui_graph,
                    t_i_image_embed,
                    t_i_text_embed,
                    t_u_image_embed,
                    t_u_text_embed,
                    users=None,
                    pos_items=None,
                    neg_items=None,
                    is_test=True,
                )

            else:

                with torch.no_grad():
                    (
                        t_u_embed,
                        t_i_embed,
                        t_i_image_embed,
                        t_i_text_embed,
                        t_u_image_embed,
                        t_u_text_embed,
                        *rest,
                    ) = self.teacher_model(
                        self.ui_graph, self.iu_graph, self.prompt_module
                    )
                    if self.student_model_type == "lightgcl":
                        u_embed, i_embed, *rest = self.student_model(
                            self.ui_graph,
                            t_i_image_embed,
                            t_i_text_embed,
                            t_u_image_embed,
                            t_u_text_embed,
                            users=None,
                            pos_items=None,
                            neg_items=None,
                            is_test=True,
                        )
                    elif self.student_model_type == "mlpgcl":
                        u_embed, i_embed = self.student_model(
                            self.ui_graph,
                            users=None,
                            pos_items=None,
                            neg_items=None,
                            is_test=True,
                        )

                    elif self.student_model_type == "lightgcn":
                        raise NotImplementedError("还没写")
                    elif self.student_model_type == "mlp":
                        raise NotImplementedError("还没写")

        result = test_torch(u_embed, i_embed, users_to_test, is_val)
        return result

    def mm(self, x, y):
        return torch.sparse.mm(x, y) if self.args.sparse else torch.mm(x, y)
