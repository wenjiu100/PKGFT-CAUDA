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
# from DA_engine_for_finetuning import train_one_epoch, evaluate
from DT import train_one_epoch, evaluate, build_source_bank
# from utils import *
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import random
# from datasets.dataset_dreamer import LoadDataset
import torch.backends.cudnn as cudnns
import modeling_finetune
import util



def get_args():
    parser = argparse.ArgumentParser('LaBraM fine-tuning and evaluation script for EEG classification', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # robust evaluation
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

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
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
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
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
    parser.add_argument('--bases-lr', default=5e-3, type=float)
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
    parser.add_argument('--seed', default=42, type=int)
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
    
    parser.add_argument('--pseudo_start_epoch', type=int, default=20)
    parser.add_argument('--lambda_pseudo', type=float, default=0.3)
    parser.add_argument('--mu_ent', type=float, default=0.1)
    parser.add_argument('--k', type=int, default=7, help='k-NN for pseudo label')
    # Mixup parameters
    parser.add_argument('--mixup_alpha', default=1.0, type=float,
                        help='Mixup的alpha参数，控制混合强度')
    parser.add_argument('--same_class_mixup', action='store_true',
                        help='是否只对同类别使用Mixup')
    parser.add_argument('--no_same_class_mixup', action='store_false', dest='same_class_mixup')
    parser.set_defaults(same_class_mixup=True)  # 默认开启只对同类别使用Mixup

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            # from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


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


def get_dataset(args, test_folder, public_folder_path):
    if args.dataset == 'SEED':
        # 调用 prepare_SEED_dataset 获取训练集、测试集和验证集
        train_dataset, test_dataset = utils.prepare_SEED_dataset(
            public_folder_path, test_folder
        )

        # 定义 EEG 通道名称（根据 SEED 数据集的通道列表）
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


        return train_dataset, test_dataset, ch_names, metrics



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

    public_folder_path = "/home/cdx/LaBraM-main/LaBraM-main/processed_data_3"  # 公共文件夹路径
    all_subfolders = [f for f in os.listdir(public_folder_path)
                      if os.path.isdir(os.path.join(public_folder_path, f))]
    all_subfolders = [os.path.join(public_folder_path, f) for f in all_subfolders]

    all_test_metrics = []
    all_best_metrics = []

    for fold, test_folder in enumerate(all_subfolders):
        print(f"Fold {fold + 1}/{len(all_subfolders)} - Test Folder: {test_folder}")
        # >>>>>>>>>> 替换为以下代码 <<<<<<<<<<
        fold_output_dir = os.path.join(args.output_dir, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)

        # 检查是否存在最终模型（best.pth）或已训练足够多的 epoch
        best_ckpt = os.path.join(fold_output_dir, "checkpoint-best.pth")
        log_file = os.path.join(fold_output_dir, "log.txt")

        is_finished = False
        if os.path.exists(best_ckpt):
            is_finished = True
        elif os.path.exists(log_file):
            # 从 log.txt 中解析最大 epoch  
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                max_epoch = -1
                for line in lines:
                    if '"epoch":' in line:
                        try:
                            data = json.loads(line.strip())
                            epoch = data.get("epoch", -1)
                            if isinstance(epoch, int):
                                max_epoch = max(max_epoch, epoch)
                        except:
                            continue
                # 如果训练到了最后一个 epoch（epochs-1），视为完成
                if max_epoch >= args.epochs - 1:
                    is_finished = True
            except Exception as e:
                print(f"Warning: Failed to parse log.txt for fold {fold}: {e}")

        if is_finished:
            print(f"Fold {fold} already completed (epoch >= {args.epochs - 1} or best model exists). Skipping...")
            # 尝试加载 best_metrics（和你原来一样）
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in reversed(lines):
                        if '"best_test_metrics":' in line:
                            try:
                                data = json.loads(line.strip())
                                best_metrics = data["best_test_metrics"]
                                all_best_metrics.append(best_metrics)
                                break
                            except:
                                pass
            continue
        # >>>>>>>>>> end <<<<<<<<<<
        

        best_metrics = {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "f1_weighted": 0.0,
            "cohen_kappa": 0.0
 
        }

        # 加载训练集和测试集
        dataset_train, dataset_test, ch_names, metrics = get_dataset(
            args, test_folder, public_folder_path
        )


        if args.disable_eval_during_finetuning:
       
            dataset_test = None

        if True:  # args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
            if args.dist_eval:                
                if type(dataset_test) == list:
                    sampler_test = [torch.utils.data.DistributedSampler(
                        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False) for dataset in dataset_test]
                else:
                    sampler_test = torch.utils.data.DistributedSampler(
                        dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            else:               
                sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
           

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

        from torch.utils.data import Dataset
        import math

        class MultipliedDataset(Dataset):
            def __init__(self, dataset, multiplier=2):
                self.dataset = dataset
                self.multiplier = multiplier
                self.length = math.ceil(len(dataset) * multiplier)

            def __len__(self):
                return self.length

            def __getitem__(self, idx):
                # 确保索引在原始数据集范围内循环
                return self.dataset[idx % len(self.dataset)]

        # 创建新的数据集
        multiplied_dataset_test = MultipliedDataset(dataset_test, multiplier=16)

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

          # 您希望冻结的层数
        # # freeze
        # main文件
        # frozen_layers = 12
        #   # 您希望冻结的层数
        # model.freeze_layers(frozen_layers)



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


                    
        data_loader_dict = {"source_loader": data_loader_train}

        source_feat_bank, source_score_bank = build_source_bank(
        data_loader_dict["source_loader"], model, device)


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
            # train_stats = train_one_epoch(
            #     model, criterion, data_loader_train, test_data, optimizer,
            #     device, epoch, loss_scaler, args.clip_grad, model_ema,
            #     log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            #     lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            #     num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            #     ch_names=ch_names, is_binary=args.nb_classes == 1,

            # )
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer,
                device, epoch, loss_scaler, args.clip_grad, model_ema,
                log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
                ch_names=ch_names, is_binary=args.nb_classes == 1,
                target_loader=test_data,  # 这里是目标域（target）数据
                source_feat_bank=source_feat_bank,
                source_score_bank=source_score_bank,
                args=args
)



            if args.output_dir and args.save_ckpt:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq)

            test_stats = evaluate(data_loader_test, model, device, header='Test:', ch_names=ch_names, metrics=metrics,
                                  is_binary=args.nb_classes == 1)
            all_test_metrics.append(test_stats)
            print(
                f"Accuracy of the network on the {len(dataset_test)} test samples: {test_stats['balanced_accuracy']:.2f}%")

            # 更新最佳模型和指标（基于测试集的准确率）
            if test_stats["accuracy"] > max_accuracy:
                best_metrics = test_stats.copy()
                max_accuracy = test_stats["accuracy"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch="best",
                        model_ema=model_ema
                    )
                # 记录测试集的最佳指标
                max_balanced_accuracy = test_stats["balanced_accuracy"]
                max_cohen_kappa = test_stats.get("cohen_kappa", 0.0)
                max_f1_weighted = test_stats.get("f1_weighted", 0.0)
                max_pr_auc = test_stats.get("pr_auc", 0.0)
                max_roc_auc = test_stats.get("roc_auc", 0.0)

            # 打印最佳指标
            print(f'Current Test Accuracy: {test_stats["accuracy"]:.4f}% | '
                  f'Best Test Accuracy: {max_accuracy:.4f}% | '
                  f'Balanced Acc: {max_balanced_accuracy:.4f}% | '
                  f'F1 Weighted: {max_f1_weighted:.4f} | '
                  f'Cohen Kappa: {max_cohen_kappa:.4f} | '
                  f'PR AUC: {max_pr_auc:.4f} | '
                  f'ROC AUC: {max_roc_auc:.4f}')

            # 日志记录（仅训练集和测试集）
            if log_writer is not None:
                # 记录训练指标
                for key, value in train_stats.items():
                    log_writer.update(f"train_{key}", value, step=epoch)
                # 记录测试指标
                for key, value in test_stats.items():
                    log_writer.update(f"test_{key}", value, step=epoch)

            # 构建日志统计字典
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters,
                'best_test_metrics': {
                    'accuracy': max_accuracy,
                    'balanced_accuracy': max_balanced_accuracy,
                    'cohen_kappa': max_cohen_kappa,
                    'f1_weighted': max_f1_weighted,
                    'pr_auc': max_pr_auc,
                    'roc_auc': max_roc_auc
                }
            }

            # 写入日志文件
            if args.output_dir and utils.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats, indent=4) + "\n")

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print(f"Total training time: {total_time_str}")
        all_best_metrics.append(best_metrics)
    # === 计算所有被试的最佳指标平均值 ===
    if all_best_metrics:
        avg_metrics = {}
        all_subject_metrics = {}
        # 遍历所有指标键（如 accuracy, balanced_accuracy 等）
        for key in all_best_metrics[0].keys():
            # 计算每个指标的平均值
            # 在计算平均值时，使用 .get() 方法处理缺失的键
            subject_values = [m.get(key, 0.0) for m in all_best_metrics]
            avg_metrics[key] = np.mean([m.get(key, 0.0) for m in all_best_metrics])

        # === 输出结果到控制台 ===
        print("\n\nDetailed Results for Each Subject:")
        # 获取所有指标的名称
        metrics_names = list(all_best_metrics[0].keys())
        # 打印表头
        header = "Subject ID\t" + "\t".join(metrics_names)
        print(header)
        # 打印分隔线
        print("-" * len(header))
        # 打印每个被试的结果
        for i, metrics in enumerate(all_best_metrics):
            row = f"Subject {i+1}"
            for key in metrics_names:
                row += f"\t{metrics.get(key, 0.0):.4f}"
            print(row)
        print("\n\nFinal Average of Best Metrics Across All Subjects:")
        print(f"Accuracy: {avg_metrics['accuracy'] * 100:.2f}%")
        print(f"Balanced Accuracy: {avg_metrics['balanced_accuracy'] * 100:.2f}%")
        print(f"F1 Weighted: {avg_metrics['f1_weighted']:.4f}")
        print(f"Cohen Kappa: {avg_metrics['cohen_kappa']:.4f}")
        # print(f"ROC AUC: {avg_metrics['roc_auc']:.4f}")
        # print(f"PR AUC: {avg_metrics['pr_auc']:.4f}")

        # === 写入日志文件 ===
        if args.output_dir and utils.is_main_process():
            log_path = os.path.join(args.output_dir, "log.txt")
            with open(log_path, "a", encoding="utf-8") as f:
                            # 写入所有被试的详细结果
                f.write("\n\n=== Detailed Results for Each Subject ===\n")
                # 写入表头
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
                # 写入每个被试的结果
                for i, metrics in enumerate(all_best_metrics):
                    row = f"Subject {i+1}"
                    for key in metrics_names:
                        row += f"\t{metrics.get(key, 0.0):.4f}"
                    f.write(row + "\n")
                f.write("\n=== Final Average of Best Metrics ===\n")
                f.write(json.dumps(avg_metrics, indent=4) + "\n")
    else:
        print("No metrics collected. Check your data or implementation.")



if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)