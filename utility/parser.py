import argparse
import os

# 文件读取保存的根路径
root_path = os.getcwd()
data_path = os.path.join(root_path, "data")

parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, default="MMTA_KD")
parser.add_argument("--dataset", type=str, default="tiktok")
parser.add_argument("--data_path", default=data_path)
parser.add_argument("--Ks", default="[10, 20, 40, 50]")
parser.add_argument("--seed", type=int, default=14322)
parser.add_argument("--sparse", type=int, default=1)
parser.add_argument("--test_flag", default="part")
parser.add_argument("--edge_mask", type=int, default=0)
parser.add_argument("--edge_mask_rate", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--cf_model", default="light_init")
parser.add_argument("--early_stopping_patience", type=int, default=8)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--regs", default="[1e-5,1e-5,1e-2]")
parser.add_argument("--emb_reg", type=float, default=1e-7)
parser.add_argument("--teacher_model_type", type=str, default="gcl")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument(
    "--teacher_model_dict_name", type=str, default="teacher_model_great"
)
parser.add_argument("--teacher_reg_rate", type=float, default=1)
parser.add_argument("--t_weight_decay", type=float, default=0.001)
parser.add_argument("--t_feat_mf_rate", type=float, default=0.001)
parser.add_argument("--feat_reg_decay", type=float, default=1e-5)
parser.add_argument("--is_softmax", type=bool, default=False)
parser.add_argument("--is_gcl_softmax", type=bool, default=False)
parser.add_argument("--teacher_assistant_model_type", type=str, default="lightgcl")
parser.add_argument(
    "--teacher_assistant_model_dict_name",
    type=str,
    default="teacher_assistant_model_great",
)
parser.add_argument("--student_model_type", type=str, default="mlpgcl")
parser.add_argument("--student_model_dict_name", type=str, default="")
parser.add_argument("--student_embed_size", type=int, default=64)
parser.add_argument("--student_lr", type=float, default=0.001)
parser.add_argument("--student_reg_rate", type=float, default=1)
parser.add_argument("--student_drop_rate", type=float, default=0.2)
parser.add_argument("--student_tau", type=float, default=5)
parser.add_argument("--embed_size", type=int, default=64)
parser.add_argument("--drop_rate", type=float, default=0.4)
parser.add_argument("--weight_size", default="[64, 64]")
parser.add_argument("--model_cat_rate", type=float, default=0.028)
parser.add_argument("--layers", type=int, default=1)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--ta_n_layers", type=int, default=1)
parser.add_argument("--student_n_layers", type=int, default=1)
parser.add_argument("--mlp_n_layers", type=int, default=1)
parser.add_argument("--if_train_teacher", type=bool, default=False)
parser.add_argument("--is_train_student", type=bool, default=False)
parser.add_argument("--kd_loss_rate", type=float, default=5e-6)
parser.add_argument("--kd_loss_feat_rate", type=float, default=0.1)
parser.add_argument("--cl_loss_rate", type=float, default=0.002)
parser.add_argument("--svd_gcl_rate", type=float, default=1.0)
parser.add_argument("--x_gcl_rate", type=float, default=1.0)
parser.add_argument("--layer_gcl", type=float, default=1.0)
parser.add_argument("--svd_layer_gcl", type=float, default=0.0)
parser.add_argument("--xsvd_gcl", type=float, default=0.0)
parser.add_argument("--x_layer_gcl", type=float, default=0.0)
parser.add_argument("--ssm_rate", type=float, default=0.6)
parser.add_argument("--s_layer_gcl", type=float, default=0.0025)
parser.add_argument("--t_cl_loss_rate", type=float, default=1e-2)
parser.add_argument("--hard_token_type", type=str, default="pca")
parser.add_argument("--soft_token_rate", type=float, default=0.1)
parser.add_argument("--feat_soft_token_rate", type=float, default=9)
parser.add_argument("--t_prompt_rate1", type=float, default=1e2)
parser.add_argument("--prompt_dropout", type=float, default=0)
parser.add_argument("--alpha_l", type=float, default=2)
parser.add_argument("--feat_loss_type", type=str, default="sce")
parser.add_argument("--neg_sample_num", type=int, default=10)
parser.add_argument("--list_wise_loss_rate", type=float, default=1)
parser.add_argument("--q", type=int, default=1)
parser.add_argument("--eps", type=float, default=0.2)
parser.add_argument("--kd_t_decay_threshold", type=float, default=0.0)
parser.add_argument("--kd_ta_decay_rate", type=float, default=6e-1)
parser.add_argument("--kd_t_decay_rate", type=float, default=6e-1)
parser.add_argument("--t_init_method", type=str, default="uniform")
parser.add_argument("--norm_mode", default="None", type=str)
parser.add_argument("--ta_norm_mode", default="None", type=str)
parser.add_argument("--s_norm_mode", default="None", type=str)
parser.add_argument("--s_norm_scale", default=5e-2, type=float)
parser.add_argument("--ta_norm_scale", default=0.0, type=float)
parser.add_argument("--norm_scale", default=8e-2, type=float)
parser.add_argument("--kd_loss_type", default="sinkhorn", type=str)
parser.add_argument("--is_teacher_kd", default=False, type=bool)
parser.add_argument("--init_teacher", default=False, type=bool)
parser.add_argument("--t_bpr_loss_rate", default=1.0, type=float)

args = parser.parse_args()


def init_args():
    return args


def save_parms(args, recall_20=0.0):
    import os
    from pprint import pformat

    root_path = os.getcwd()
    data_path = os.path.join(root_path, "data")

    max_key_length = max(len(key) for key in vars(args).keys())
    args.recall_20 = recall_20

    ordered_keys = [
        "student_lr",
        "dataset",
        "seed",
        "cl_loss_rate",
        "kd_loss_rate",
        "kd_loss_rate_sink",
        "kd_loss_feat_rate",
        "kd_loss_list_rate",
    ] + sorted(
        set(vars(args).keys())
        - {
            "student_lr",
            "dataset",
            "seed",
            "cl_loss_rate",
            "kd_loss_rate",
            "kd_loss_rate_sink",
            "kd_loss_feat_rate",
            "kd_loss_list_rate",
        }
    )

    formatted_args = {
        key: f"{' ' * (max_key_length - len(key))}{value}"
        for key, value in vars(args).items()
    }

    formatted_str = pformat(formatted_args)
    output_folder = os.path.join(root_path, "output")
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, "args_output.txt")

    with open(output_file, "a") as file:
        file.write("\n\n")
        file.write(
            f"## {args.dataset}_{args.seed}_{args.recall_20}_{args.student_model_type}"
        )
        file.write("\n\n" + formatted_str + "\n\n")

    print(f"Arguments have been written to {output_file}")


def format_args(args):
    args_dict = vars(args)
    max_key_length = max(len(key) for key in args_dict.keys())
    formatted_lines = [
        f"{key.ljust(max_key_length)} : {value}" for key, value in args_dict.items()
    ]
    return "\n" + "\n".join(formatted_lines)


def select_dataset():
    pass
