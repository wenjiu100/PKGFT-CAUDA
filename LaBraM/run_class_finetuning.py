# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
import argparse
import datetime
from pyexpat import model
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
from collections import OrderedDict
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from DA_engine_for_finetuning import train_one_epoch, evaluate
# from engine_for_finetuning import train_one_epoch, evaluate 
# from utils import *
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import random
# from datasets.dataset_dreamer import LoadDataset
import torch.backends.cudnn as cudnns
import modeling_finetune
import util


def get_args(seed=None):
    parser = argparse.ArgumentParser('LaBraM fine-tuning and evaluation script for EEG classification', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # robust evaluation  88
    parser.add_argument('--robust_test', default=None, type=str,
                        help='robust evaluation dataset')

    # Model parameters
    parser.add_argument('--model', default='labram_base_patch200_200', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=True)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=200, type=int,
                        help='EEG input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--use_iba', action='store_true', default=True,
                        help='Use IntraBlockAdapter')
    parser.add_argument('--use_cba', action='store_true', default=True,
                        help='Use CrossBlockAdapter')
    parser.add_argument('--cba_stride', type=int, default=6,
                        help='Stride for CrossBlockAdapter')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR 5, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Finetuning params
    parser.add_argument('--finetune', default='./checkpoints/labram-base.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)
    parser.add_argument('--bases-scale', default=1., type=float, help='bases alpha/bases')
    parser.add_argument('--bases_coeff_reg', default='None', choices=[None, 'rank', 'ortho'])
    parser.add_argument('--bases-lr', default=5e-4, type=float)
    parser.add_argument('--bases-decay', default=1e-6, type=float)
    parser.add_argument('--reg_lambda', default=0.01, type=float)
    # finetuning
    parser.add_argument('--tuning-mode', default= 'full', type=str,
                        help='Method of fine-tuning (default: None')
    
          
    # Dataset parameters
    parser.add_argument('--datasets_dir', type=str, default='datasets_dir', help='datasets_dir')
    parser.add_argument('--nb_classes', default=0, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=2025, type=int)
    # parser.add_argument('--seed', default=42, type=int)
    # parser.add_argument('--seed', default=0, type=int)
    #parser.add_argument('--seed', default=420, type=int)
    #parser.add_argument('--seed', default=2026, type=int)

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')  
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    parser.set_defaults(pin_mem=True)
    # lora
    parser.add_argument('--lora_dim', type=int, default=8, help='lora attn dimension')
    parser.add_argument('--lora_alpha', type=int, default=128, help='lora attn alpha')
    parser.add_argument('--lora_dropout', default=0.1, type=float, help='dropout probability for lora layers')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--dataset', default='SEED', type=str,
                        help='dataset: TUAB | TUEV | SEED | SLEEP')
    # Mixup parameters
    parser.add_argument('--mixup_alpha', default=1.0, type=float,
                        help='Mixup的alpha参数,控制混合强度')
    parser.add_argument('--same_class_mixup', action='store_true',
                        help='是否只对同类别使用Mixup')
    parser.add_argument('--no_same_class_mixup', action='store_false', dest='same_class_mixup')
    parser.set_defaults(same_class_mixup=True)  # 默认开启只对同类别使用Mixup

    known_args, _ = parser.parse_known_args()

    if seed is not None:
        known_args.seed = seed

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    # return parser.parse_args(), ds_init
    return known_args, ds_init



def get_models(args):
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        qkv_bias=args.qkv_bias,
        tuning_mode=args.tuning_mode,
        attn_dcf_scale=args.bases_scale
    )

    return model


def get_dataset(args):
    if args.dataset == 'TUAB':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUAB_dataset("/home/cdx/LaBraM-main/LaBraM-main/v3.0.1/edf/processed")
        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                    'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF',
                    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
    elif args.dataset == 'TUEV':
        train_dataset, test_dataset, val_dataset = utils.prepare_TUEV_dataset("/home/cdx/LaBraM-main/LaBraM-main/v2.0.1/edf/processed")
        ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                    'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF',
                    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 6
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    elif args.dataset == 'SEED':
        train_dataset, test_dataset, val_dataset = utils.prepare_SEED_dataset("/home/cdx/LaBraM-main/LaBraM-main/datasets_dir/processed1")
        ch_names = ['FP1', 'FPZ', 'FP2',
                    'AF3', 'AF4', \
                    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', \
                    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', \
                    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', \
                    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', \
                    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', \
                    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', \
                    'CB1', 'O1', 'OZ', 'O2', 'CB2']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 5
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    return train_dataset, test_dataset, val_dataset, ch_names, metrics



def calculate_model_metrics(model, input_size=(1, 23, 10, 200)):
    from fvcore.nn import FlopCountAnalysis, parameter_count
    import torch
    import types

    actual_model = model.module if hasattr(model, 'module') else model

    # === 清理干扰属性 ===
    for attr in ['pos_embed_ft', 'pos_embed_highres']:
        if hasattr(actual_model, attr):
            delattr(actual_model, attr)
    if hasattr(actual_model, 'interpolate_pos_encoding'):
        actual_model.interpolate_pos_encoding = False

    base_pos_len = actual_model.pos_embed.shape[1]  # 129
    num_patches = base_pos_len - 1                   # 128

    B, C_orig, _, P = input_size
    if num_patches % C_orig == 0:
        C, T = C_orig, num_patches // C_orig
    else:
        for c in [64, 32, 16, 128, 20, 40, 80, 23]:
            if num_patches % c == 0:
                C, T = c, num_patches // c
                break
        else:
            C, T = 1, num_patches

    input_tensor = torch.randn(B, C, T, P, device=next(model.parameters()).device)
    print(f"[INFO] Input tensor shape: {input_tensor.shape}")

    # === 保存原始 forward ===
    original_forward = actual_model.forward

    # === 定义安全 forward：只修改 pos_embed 使用方式 ===
    def safe_forward(self, x, **kwargs):
        # Step 1: patch embedding (same as original)
        x = self.patch_embed(x)  # (B, N, D), N = C*T = num_patches

        # Step 2: add cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 129, D)

        # ⭐⭐⭐ 关键修复：强制使用 self.pos_embed，不管其他逻辑 ⭐⭐⭐
        pos_embed = self.pos_embed  # (1, 129, D)
        if pos_embed.shape[1] != x.shape[1]:
            raise RuntimeError(f"pos_embed {pos_embed.shape} vs x {x.shape}")
        x = x + pos_embed

        # Step 3: transformer blocks
        for blk in self.blocks:
            x = blk(x)
        if self.norm is not None:
            x = self.norm(x)

        # Step 4: head
        return self.head(x[:, 0])

    # 绑定新 forward
    actual_model.forward = types.MethodType(safe_forward, actual_model)

    try:
        with torch.no_grad():
            flops = FlopCountAnalysis(model, input_tensor)
            total_flops = flops.total()
            total_params = parameter_count(model)['']
    finally:
        # 恢复原始 forward
        actual_model.forward = original_forward

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n✅ Success! FLOPs and params calculated.\n")
    print(f"FLOPs: {total_flops / 1e9:.2f} GFLOPs")
    print(f"Total Params: {total_params / 1e6:.2f}M")
    print(f"Trainable Params: {trainable_params / 1e6:.2f}M")

    return {'flops': total_flops, 'total_params': total_params, 'trainable_params': trainable_params}


def plot_cls_attention_per_layer(avg_attention_to_cls, save_path=None):
    """
    avg_attention_to_cls: np.ndarray [L, H]
    """
    import numpy as np
    import matplotlib.pyplot as plt

    cls_attn = avg_attention_to_cls.mean(axis=1)
    layers = np.arange(len(cls_attn))
    mean = cls_attn.mean()

    plt.figure(figsize=(8, 5))

    plt.plot(
        layers,
        cls_attn,
        marker='o',
        linewidth=2,
        label='CLS Attention (per layer)'
    )

    plt.axhline(
        mean,
        color='r',
        linestyle='--',
        linewidth=2,
        label=f'Global Mean = {mean:.4f}'
    )

    # ✅ 固定 y 轴（关键）
    plt.ylim(0.0, 0.02)


    # ymin = cls_attn.min()
    # ymax = cls_attn.max()

    # # 加 10% 边距，防止贴边
    # margin = 0.1 * (ymax - ymin)

    # plt.ylim(
    #     max(0.0, ymin - margin),
    #     ymax + margin
    # )

    plt.xlabel('Transformer Layer')
    plt.ylabel('CLS (First Token) Attention Score')
    plt.title('CLS Attention Across Transformer Layers (EEG)')
    plt.grid(alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✅ 图1已保存到 {save_path}")

    plt.show()


def plot_qk_attention_heatmap(
    attn,          # [N, N]  (already averaged)
    layer_idx=None,
    save_path=None,
    title=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    assert attn.ndim == 2, f"Expected [N, N], got {attn.shape}"

    # 1️⃣ Row-wise normalization（EEG 推荐）
    attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-6)

    # 2️⃣ Percentile clipping（防止 EEG 极值）
    vmin = np.percentile(attn, 1)
    vmax = np.percentile(attn, 99)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        attn,
        cmap='viridis',
        square=True,
        vmin=vmin,
        vmax=vmax,
        cbar=True
    )

    plt.xlabel('Key Token Index')
    plt.ylabel('Query Token Index')

    if title is None:
        if layer_idx is not None:
            title = f'Q–K Attention Heatmap (Layer {layer_idx + 1})'
        else:
            title = 'Q–K Attention Heatmap'

    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"✅ 图已保存到 {save_path}")

    plt.show()





# def plot_qk_attention_heatmap(attn_matrix, save_path="qk_attention_heatmap.png"):
#     """
#     attn_matrix: np.ndarray [N, N]  (query × key)
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     # 1️⃣ Row-wise normalization（每个 query 内部对比）
#     attn = attn_matrix / (attn_matrix.sum(axis=-1, keepdims=True) + 1e-6)

#     # 2️⃣ 分位裁剪（避免极端值）
#     vmax = np.percentile(attn, 99)
#     vmin = np.percentile(attn, 1)

#     plt.figure(figsize=(7, 6))
#     sns.heatmap(
#         attn,
#         cmap='viridis',
#         square=True,
#         cbar=True,
#         vmin=vmin,
#         vmax=vmax
#     )

#     plt.xlabel('Key Token Index')
#     plt.ylabel('Query Token Index')
#     plt.title('Query–Key Attention Heatmap (EEG)')

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     plt.close()

#     print(f"✅ 图2已保存到 {save_path}")



def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # dataset_train, dataset_test, dataset_val: follows the standard format of torch.utils.data.Dataset.
    # ch_names: list of strings, channel names of the dataset. It should be in capital letters.
    # metrics: list of strings, the metrics you want to use. We utilize PyHealth to implement it.
    dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(args)
    print("Training dataset:", dataset_train)

    if args.disable_eval_during_finetuning:
        dataset_val = None
        dataset_test = None

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            if type(dataset_test) == list:
                sampler_test = [torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False) for dataset in dataset_test]
            else:
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        if type(dataset_test) == list:
            data_loader_test = [torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            ) for dataset, sampler in zip(dataset_test, sampler_test)]
        else:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
    else:
        data_loader_val = None
        data_loader_test = None


    from torch.utils.data import Dataset
    import math

    class MultipliedDataset(Dataset):
        def __init__(self, dataset, multiplier=1.5):
            self.dataset = dataset
            self.multiplier = multiplier
            self.length = math.ceil(len(dataset) * multiplier)

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            # 确保索引在原始数据集范围内循环
            return self.dataset[idx % len(self.dataset)]

    # 创建新的数据集
    multiplied_dataset_test = MultipliedDataset(dataset_test, multiplier=3.5)

    # 验证数据集大小
    original_size = len(dataset_test)
    multiplied_size = len(multiplied_dataset_test)
    
    print(f"Original dataset size: {original_size}")
    print(f"Multiplied dataset size: {multiplied_size}")
    Msampler_test = torch.utils.data.RandomSampler(multiplied_dataset_test)

    # 使用新的数据集创建 DataLoader
    test_data = torch.utils.data.DataLoader(
        multiplied_dataset_test,
        sampler=Msampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )



    model = get_models(args)


    
 
      # 您希望冻结的层数s
    # # freeze
    # main文件
    # for i in range(13):   
    # frozen_layers = 1
    # print("frozen_layers",frozen_layers)
    # # #   # 您希望冻结的层数
    # model.freeze_layers(frozen_layers)


    # 假设你的模型实例为 model
    # model.freeze_layers(11)

    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (args.model_filter_name != ''):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    # calculate_model_metrics(model, input_size=(1, 23, 10, 200))

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if args.nb_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        balanced_accuracy = []
        accuracy = []
        for data_loader in data_loader_test:
            test_stats = evaluate(data_loader, model, device, header='Test:', ch_names=ch_names, metrics=metrics,
                                is_binary=(args.nb_classes == 1))
            accuracy.append(test_stats['accuracy'])
            balanced_accuracy.append(test_stats['balanced_accuracy'])
        print(
            f"======Accuracy: {np.mean(accuracy)} {np.std(accuracy)}, balanced accuracy: {np.mean(balanced_accuracy)} {np.std(balanced_accuracy)}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_accuracy_test = 0.0
    relative_norms = [] 
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        # 这个方法会消耗迭代器，因此只适合在初始化时使用。
        # train_count = sum(1 for _ in data_loader)
        # test_count = sum(1 for _ in test_data)
        # # print(f"Training dataset has {train_count} batches.")
        # print(f"Test dataset has {test_count} batches.")

        print("Data loader length:", len(data_loader_train))


        train_stats = train_one_epoch(
            model, criterion, data_loader_train, data_loader_test, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            ch_names=ch_names, is_binary=args.nb_classes == 1
        )


        if args.output_dir and args.save_ckpt:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq)

        if data_loader_val is not None:
            val_stats = evaluate(data_loader_val, model, device, header='Val:', ch_names=ch_names, metrics=metrics,
                                is_binary=args.nb_classes == 1)
            print(f"Accuracy of the network on the {len(dataset_val)} val EEG: {val_stats['balanced_accuracy']:.2f}%")

            # test_stats, M_Act, F_Attn = evaluate(data_loader_test, model, device, header='Test:', ch_names=ch_names, metrics=metrics,
            #                     is_binary=args.nb_classes == 1, return_attention=True)
            test_stats = evaluate(data_loader_test, model, device, header='Test:', ch_names=ch_names, metrics=metrics,
                    is_binary=args.nb_classes == 1)
            print(f"Accuracy of the network on the {len(dataset_test)} test EEG: {test_stats['balanced_accuracy']:.2f}%")

            if max_accuracy < val_stats["accuracy"]:
                max_accuracy = val_stats["accuracy"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                max_accuracy_test = test_stats["accuracy"]
                max_balanced_accuracy = test_stats["balanced_accuracy"]
                max_cohen_kappa = test_stats["cohen_kappa"]
                max_f1_weighted = test_stats["f1_weighted"]
                # max_pr_auc = test_stats["pr_auc"]
                # max_roc_auc = test_stats["roc_auc"]
                # 🔥 关键：只有当这是新的 best，并且 epoch >= 10，才生成 t-SNE
                # if epoch >=2:
                #     try:
                #         print(f"🎨 Generating t-SNE plot for BEST epoch {epoch}...")
                #         visualize_tsne(
                #             model=model,
                #             data_loader=data_loader_test,
                #             device=device,
                #             ch_names=ch_names,
                #             output_dir="tsne_results",
                #             epoch=epoch
                #         )
                #         best_epoch_for_tsne = epoch  # 记录
                #     except Exception as e:
                #         print(f"⚠️ Failed to generate t-SNE at best epoch {epoch}: {e}")  
                # plot_cls_attention_per_layer(
                #         avg_attention_to_cls,
                #         save_path="cls_attention_per_layer.png"
                #     )  

                # plot_qk_attention_heatmap(
                #     qk_attentions[9],
                #     layer_idx=9,
                #     save_path="qk_layer9.png"
                # )

                # plot_qk_attention_heatmap(
                #     qk_attentions[10],
                #     layer_idx=10,
                #     save_path="qk_layer10.png"
                # )


            # print(f'Max accuracy val: {max_accuracy:.2f}%, max accuracy test: {max_accuracy_test:.2f}%')
            print(f'Max accuracy val: {max_accuracy:.4f}%, '
                f'max accuracy test: {max_accuracy_test:.4f}%, '
                f'max balanced accuracy: {max_balanced_accuracy:.4f}%, '
                f'max cohen kappa: {max_cohen_kappa:.4f}%, '
                f'max f1 weighted: {max_f1_weighted:.4f}%, '
                # f'max pr auc: {max_pr_auc:.4f}, '
                # f'max roc auc: {max_roc_auc:.4f}'
                )
            # print("M-Act':", M_Act)
            # print("F-Attn:", F_Attn)

            if log_writer is not None:
                for key, value in val_stats.items():
                    if key == 'accuracy':
                        log_writer.update(accuracy=value, head="val", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(balanced_accuracy=value, head="val", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value, head="val", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(pr_auc=value, head="val", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(roc_auc=value, head="val", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value, head="val", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="val", step=epoch)
                for key, value in test_stats.items():
                    if key == 'accuracy':
                        log_writer.update(accuracy=value, head="test", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(balanced_accuracy=value, head="test", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value, head="test", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(pr_auc=value, head="test", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(roc_auc=value, head="test", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value, head="test", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="test", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'val_{k}': v for k, v in val_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    import random
    from pathlib import Path
    # opts, ds_init = get_args()
    # if opts.output_dir:
    #     Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    # main(opts, ds_init)
    seeds = [2025, 420, 42, 0, 2026]  # 可以改成你想跑的 seed 列表
    base_output_dir = "results"  # 总输出目录
    base_log_dir = "logs"        # 总日志目录

    for seed_val in seeds:
        print(f"\n===== Running seed {seed_val} =====\n")
        
        # 1️⃣ 获取 args，并传入 seed
        opts, ds_init = get_args(seed=seed_val)

        # 2️⃣ 🔥 关键：多 seed 实验，强制关闭 auto_resume
        opts.auto_resume = False
        opts.resume = ''   # 双保险，防止误 resume

        # 2️⃣ 设置随机种子
        seed = opts.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 3️⃣ 设置每个 seed 的输出目录和日志目录
        opts.output_dir = os.path.join(base_output_dir, f"seed_{seed_val}")
        opts.log_dir = os.path.join(base_log_dir, f"seed_{seed_val}")
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
        Path(opts.log_dir).mkdir(parents=True, exist_ok=True)

        # 4️⃣ 运行训练主函数
        main(opts, ds_init)