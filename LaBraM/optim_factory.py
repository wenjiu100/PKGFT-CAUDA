# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
import torch
from torch import optim as optim

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

import json

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None, **kwargs):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(kwargs.get('filter_name', [])) > 0:
            flag = False
            for filter_n in kwargs.get('filter_name', []):
                if filter_n in name:
                    print(f"filter {name} because of the pattern {filter_n}")
                    flag = True
            if flag:
                continue
        if param.ndim <= 1 or name.endswith(".bias") or name in skip_list: # param.ndim <= 1 len(param.shape) == 1
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


# def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None, **kwargs):
#     opt_lower = args.opt.lower()
#     weight_decay = args.weight_decay
#     if weight_decay and filter_bias_and_bn:
#         skip = {}
#         if skip_list is not None:
#             skip = skip_list
#         elif hasattr(model, 'no_weight_decay'):
#             skip = model.no_weight_decay()
#         print(f"Skip weight decay name marked in model: {skip}")
#         parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale, **kwargs)
#         weight_decay = 0.
#     else:
#         parameters = model.parameters()

#     if 'fused' in opt_lower:
#         assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

#     opt_args = dict(lr=args.lr, weight_decay=weight_decay)
#     if hasattr(args, 'opt_eps') and args.opt_eps is not None:
#         opt_args['eps'] = args.opt_eps
#     if hasattr(args, 'opt_betas') and args.opt_betas is not None:
#         opt_args['betas'] = args.opt_betas
    
#     print('Optimizer config:', opt_args)
#     opt_split = opt_lower.split('_')
#     opt_lower = opt_split[-1]
#     if opt_lower == 'sgd' or opt_lower == 'nesterov':
#         opt_args.pop('eps', None)
#         optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
#     elif opt_lower == 'momentum':
#         opt_args.pop('eps', None)
#         optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
#     elif opt_lower == 'adam':
#         optimizer = optim.Adam(parameters, **opt_args)
#     elif opt_lower == 'adamw':
#         optimizer = optim.AdamW(parameters, **opt_args)
#     elif opt_lower == 'nadam':
#         optimizer = Nadam(parameters, **opt_args)
#     elif opt_lower == 'radam':
#         optimizer = RAdam(parameters, **opt_args)
#     elif opt_lower == 'adamp':
#         optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
#     elif opt_lower == 'sgdp':
#         optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
#     elif opt_lower == 'adadelta':
#         optimizer = optim.Adadelta(parameters, **opt_args)
#     elif opt_lower == 'adafactor':
#         if not args.lr:
#             opt_args['lr'] = None
#         optimizer = Adafactor(parameters, **opt_args)
#     elif opt_lower == 'adahessian':
#         optimizer = Adahessian(parameters, **opt_args)
#     elif opt_lower == 'rmsprop':
#         optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
#     elif opt_lower == 'rmsproptf':
#         optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
#     elif opt_lower == 'nvnovograd':
#         optimizer = NvNovoGrad(parameters, **opt_args)
#     elif opt_lower == 'fusedsgd':
#         opt_args.pop('eps', None)
#         optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
#     elif opt_lower == 'fusedmomentum':
#         opt_args.pop('eps', None)
#         optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
#     elif opt_lower == 'fusedadam':
#         optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
#     elif opt_lower == 'fusedadamw':
#         optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
#     elif opt_lower == 'fusedlamb':
#         optimizer = FusedLAMB(parameters, **opt_args)
#     elif opt_lower == 'fusednovograd':
#         opt_args.setdefault('betas', (0.95, 0.98))
#         optimizer = FusedNovoGrad(parameters, **opt_args)
#     else:
#         assert False and "Invalid optimizer"
#         raise ValueError

#     if len(opt_split) > 1:
#         if opt_split[0] == 'lookahead':
#             optimizer = Lookahead(optimizer)

#     return optimizer

def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None, **kwargs):
    """
    创建优化器，并支持基于 tuning_mode 的参数筛选（逻辑冻结）。
    """
    # ------------------ 新增：参数筛选逻辑 ------------------
    if hasattr(args, 'tuning_mode') and args.tuning_mode:
        tuning_mode = args.tuning_mode
        print(f"Applying parameter freezing based on tuning_mode: {tuning_mode}")
        for name, param in model.named_parameters():
            # 默认所有参数都需要冻结，除非满足特定条件
            param.requires_grad = False

            # 根据不同的 tuning_mode 解冻对应的参数
            if tuning_mode == 'linear_probe':
                if "head." in name:
                    param.requires_grad = True
            elif tuning_mode == 'ssf':
                if "head." in name or "bases" in name or "ssf_scale" in name or "ssf_shift_"   in name:
                    param.requires_grad = True
            elif tuning_mode == 'dcf':
                # 解冻分类头、DCFs的bases、ssf参数
                # if ("head." in name or 
                #     "bases" in name or 
                #     "ssf_scale" in name or 
                #     "ssf_shift_" in name or
                #     "cls_token" in name or 
                #     "pos_embed" in name or 
                #     "time_embed" in name or 
                #     "patch_embed" in name or 
                #     "gamma_1" in name or 
                #     "gamma_2" in name               
                #     ):
                if "head." in name or "bases" in name or "ssf_scale" in name or "ssf_shift_"  in name or "blocks.11."  in name:
                    param.requires_grad = True
            elif tuning_mode == 'attn_dcf_vo' or tuning_mode == 'attn_dcf_kv':
                # 解冻分类头和ssf参数
                if "head." in name or "ssf_scale" in name or "ssf_shift_" in name:
                    param.requires_grad = True
            elif tuning_mode == 'ssf_cnn':
                # 解冻分类头和CNN的sc_scale参数
                if "head." in name or "sc_scale" in name:
                    param.requires_grad = True
            # 可以在这里添加更多 tuning_mode 的逻辑
            elif tuning_mode == 'full':
                # 解冻分类头和CNN的sc_scale参数
                # if ("head." in name or 
                #     "cls_token" in name or 
                #     "pos_embed" in name or 
                #     "time_embed" in name or 
                #     "patch_embed" in name or 
                #     "gamma_1" in name or 
                #     "gamma_2" in name               
                #     ):
                #     param.requires_grad = True
                for name, param in model.named_parameters():
                    param.requires_grad = True
            elif tuning_mode == 'freeze_attn_mlp':
                for name, param in model.named_parameters():
                    # 默认全部解冻
                    param.requires_grad = True
                    # 但如果是 attn 或 mlp，则冻结
                    if 'attn.' in name or 'mlp.' in name:
                        param.requires_grad = False
            # 可以在这里添加更多 tuning_mode 的逻辑
            elif tuning_mode == 'freeze_attn':
                # 冻结所有 attn；仅冻结前 5 层的 mlp
                for name, param in model.named_parameters():
                    param.requires_grad = True  # 默认解冻

                    # 检查是否属于 attention：只要包含 '.attn.' 就冻结
                    if '.attn.' in name:
                        param.requires_grad = False

                    # 检查是否属于 mlp，并且在前 5 层（layer index < 5）
                    elif '.mlp.' in name:
                        # 尝试从参数名中提取 layer index
                        # 假设命名格式如: xxx.h.3.mlp.xxx 或 xxx.layers.2.mlp.xxx
                        # 我们找 '.mlp.' 前面的数字
                        try:
                            # 找到 '.mlp.' 的位置
                            mlp_idx = name.find('.mlp.')
                            # 在 '.mlp.' 之前的部分，按 '.' 分割
                            prefix = name[:mlp_idx]
                            parts = prefix.split('.')
                            # 从后往前找第一个纯数字，作为 layer index
                            layer_index = None
                            for part in reversed(parts):
                                if part.isdigit():
                                    layer_index = int(part)
                                    break
                            # 如果成功提取到 layer_index 且 < 5，则冻结
                            if layer_index is not None and layer_index < 6:
                                param.requires_grad = False
                            # 否则（index >=5 或无法解析），保持可训练（已设为 True）
                        except Exception:
                            # 如果解析失败，默认不解冻（保守策略）
                            pass
            else:
                # 如果是未知的 tuning_mode，可以选择不解冻任何参数或解冻所有（警告）
                print(f"Warning: Unknown tuning_mode '{tuning_mode}'. Freezing all parameters or consider adding logic.")
                # 如果需要默认解冻所有，可以取消下面这行的注释
                # param.requires_grad = True 
            
            # 调试：打印被解冻的参数
            # if param.requires_grad:
            #     print(f"Unfrozen parameter: {name}")

        print('Parameter freezing based on tuning_mode finished!')
        # 可选：打印需要优化的参数数量
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_trainable_params}")
    # --------------------------------------------------------

    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    
    # --- 修改：根据是否使用 tuning_mode 和 filter_bias_and_bn 决定参数分组方式 ---
    parameters = model.parameters() # 默认
    if hasattr(args, 'tuning_mode') and args.tuning_mode:
        # 如果使用了 tuning_mode，我们依赖 requires_grad 进行筛选。
        # get_parameter_groups 会自动跳过 requires_grad=False 的参数。
        # 因此，即使 filter_bias_and_bn 为 True，我们也调用 get_parameter_groups 来应用 weight decay 逻辑。
        # skip_list 和 layer-wise decay 逻辑也可以在此应用。
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        print(f"Skip weight decay name marked in model or provided: {list(skip)}")
        
        # 调用 get_parameter_groups，它会处理 requires_grad, skip_list, filter_bias_and_bn 逻辑
        parameters = get_parameter_groups(
            model, 
            weight_decay=weight_decay, 
            skip_list=skip, 
            get_num_layer=get_num_layer, 
            get_layer_scale=get_layer_scale,
            **kwargs # 传递 filter_name 等其他参数
        )
        # 由于 get_parameter_groups 已经处理了 weight_decay，将其设为 0 以避免优化器重复应用
        weight_decay = 0.0 
    elif weight_decay and filter_bias_and_bn:
        # 如果没有使用 tuning_mode 但需要 filter_bias_and_bn，使用原始逻辑
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        print(f"Skip weight decay name marked in model or provided: {list(skip)}")
        
        parameters = get_parameter_groups(
            model, 
            weight_decay=weight_decay, 
            skip_list=skip, 
            get_num_layer=get_num_layer, 
            get_layer_scale=get_layer_scale,
            **kwargs
        )
        weight_decay = 0.0
    else:
        # 最简单的情况：所有 requires_grad=True 的参数，使用统一的 weight_decay
        parameters = model.parameters()
    # -------------------------------------------------------------------

    # --- 新增：为 'attn_dcf' 模式处理不同的学习率和权重衰减 ---
    # 注意：这部分逻辑与上面的参数分组逻辑可能冲突。
    # 如果 get_parameter_groups 已经处理了参数分组，这部分可能不需要或需要调整。
    # 假设如果 tuning_mode 是 'attn_dcf'，我们希望覆盖 get_parameter_groups 的分组，
    # 或者 get_parameter_groups 的输出已经是期望的格式。
    # 这里我们假设 get_parameter_groups 没有处理 'attn_dcf' 的特殊情况，
    # 我们需要在这里重新定义参数组。
    # 为了兼容性，我们检查 parameters 是否已经是列表（即 get_parameter_groups 的输出），
    # 如果不是，我们才应用 'attn_dcf' 的特殊分组。
    parameters_for_optimizer = parameters # 默认使用上面定义的 parameters
    if hasattr(args, 'tuning_mode') and args.tuning_mode == 'attn_dcf':
        # 只有在 parameters 还是 generator 时才应用特殊分组
        # 如果 get_parameter_groups 已经处理了，它会是 list
        if not isinstance(parameters, list): 
             # 检查 args 是否有 bases_lr 和 bases_decay 属性
            bases_lr = getattr(args, 'bases_lr', args.lr) # 默认使用主 lr
            bases_decay = getattr(args, 'bases_decay', 0.0) # 默认使用 0 weight_decay
            
            # 定义参数组 (基于 requires_grad=True 的参数)
            parameters_groups = [
                # DCF bases 参数组
                {'params': [p for n, p in model.named_parameters() if 'bases' in n and p.requires_grad],
                 'lr': bases_lr, 'weight_decay': bases_decay},
                # 其他非 bias 参数组
                {'params': [p for n, p in model.named_parameters() if 'bases' not in n and 'bias' not in n and p.requires_grad],
                 'lr': args.lr, 'weight_decay': args.weight_decay},
                # Bias 参数组 (通常 weight_decay=0)
                {'params': [p for n, p in model.named_parameters() if 'bias' in n and p.requires_grad],
                 'lr': args.lr, 'weight_decay': 0.},
            ]
            parameters_for_optimizer = parameters_groups
            # 不再需要从 opt_args 中移除 lr 和 weight_decay，因为它们可能仍用于其他参数组
            # 或者，如果所有参数都在组里定义了，可以移除
            # opt_args.pop('lr', None) 
            # opt_args.pop('weight_decay', None) 
            print(f"Using custom parameter groups for 'attn_dcf' mode with bases_lr={bases_lr}, bases_decay={bases_decay}")
        else:
            print("Parameter groups already defined by get_parameter_groups, skipping custom 'attn_dcf' grouping.")
    # ---------------------------------------------------------------

    if 'fused' in opt_lower:
        # assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'
        try:
            import apex
            has_apex = True
        except ImportError:
            has_apex = False
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    print('Optimizer config:', opt_args)
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    
    # --- 使用可能经过参数组处理的 parameters_for_optimizer ---
    try:
        if opt_lower == 'sgd' or opt_lower == 'nesterov':
            opt_args.pop('eps', None)
            optimizer = optim.SGD(parameters_for_optimizer, momentum=args.momentum, nesterov=True, **opt_args)
        elif opt_lower == 'momentum':
            opt_args.pop('eps', None)
            optimizer = optim.SGD(parameters_for_optimizer, momentum=args.momentum, nesterov=False, **opt_args)
        elif opt_lower == 'adam':
            optimizer = optim.Adam(parameters_for_optimizer, **opt_args)
        elif opt_lower == 'adamw':
            optimizer = optim.AdamW(parameters_for_optimizer, **opt_args) # <-- 使用 parameters_for_optimizer
        elif opt_lower == 'nadam':
            # 尝试 PyTorch >= 1.10 的原生 NAdam
            try:
                optimizer = optim.Nadam(parameters_for_optimizer, **opt_args)
            except AttributeError:
                 # 如果没有，需要导入自定义 Nadam
                 # from timm.optim import Nadam # 假设如此
                 optimizer = Nadam(parameters_for_optimizer, **opt_args)
        elif opt_lower == 'radam':
            # from timm.optim import RAdam
            optimizer = RAdam(parameters_for_optimizer, **opt_args)
        elif opt_lower == 'adamp':
            # from timm.optim import AdamP
            optimizer = AdamP(parameters_for_optimizer, wd_ratio=0.01, nesterov=True, **opt_args)
        elif opt_lower == 'sgdp':
            # from timm.optim import SGDP
            optimizer = SGDP(parameters_for_optimizer, momentum=args.momentum, nesterov=True, **opt_args)
        elif opt_lower == 'adadelta':
            optimizer = optim.Adadelta(parameters_for_optimizer, **opt_args)
        elif opt_lower == 'adafactor':
            # from timm.optim import Adafactor
            if not args.lr:
                opt_args['lr'] = None
            optimizer = Adafactor(parameters_for_optimizer, **opt_args)
        elif opt_lower == 'adahessian':
            # from timm.optim import Adahessian
            optimizer = Adahessian(parameters_for_optimizer, **opt_args)
        elif opt_lower == 'rmsprop':
            optimizer = optim.RMSprop(parameters_for_optimizer, alpha=0.9, momentum=args.momentum, **opt_args)
        elif opt_lower == 'rmsproptf':
            # from timm.optim import RMSpropTF
            optimizer = RMSpropTF(parameters_for_optimizer, alpha=0.9, momentum=args.momentum, **opt_args)
        elif opt_lower == 'nvnovograd':
            # from timm.optim import NvNovoGrad
            optimizer = NvNovoGrad(parameters_for_optimizer, **opt_args)
        elif opt_lower == 'fusedsgd':
            opt_args.pop('eps', None)
            optimizer = FusedSGD(parameters_for_optimizer, momentum=args.momentum, nesterov=True, **opt_args)
        elif opt_lower == 'fusedmomentum':
            opt_args.pop('eps', None)
            optimizer = FusedSGD(parameters_for_optimizer, momentum=args.momentum, nesterov=False, **opt_args)
        elif opt_lower == 'fusedadam':
            optimizer = FusedAdam(parameters_for_optimizer, adam_w_mode=False, **opt_args)
        elif opt_lower == 'fusedadamw':
            optimizer = FusedAdam(parameters_for_optimizer, adam_w_mode=True, **opt_args)
        elif opt_lower == 'fusedlamb':
            optimizer = FusedLAMB(parameters_for_optimizer, **opt_args)
        elif opt_lower == 'fusednovograd':
            opt_args.setdefault('betas', (0.95, 0.98))
            optimizer = FusedNovoGrad(parameters_for_optimizer, **opt_args)
        else:
            assert False and f"Invalid optimizer: {opt_lower}"
            raise ValueError(f"Invalid optimizer: {opt_lower}")
    except Exception as e:
        print(f"Error creating optimizer {opt_lower}: {e}")
        raise e

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            # from lookahead import Lookahead # 假设如此
            optimizer = Lookahead(optimizer)

    return optimizer
