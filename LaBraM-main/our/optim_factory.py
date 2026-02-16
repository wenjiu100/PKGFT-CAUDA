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

def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None, **kwargs):

    if hasattr(args, 'tuning_mode') and args.tuning_mode:
        tuning_mode = args.tuning_mode
        print(f"Applying parameter freezing based on tuning_mode: {tuning_mode}")
        for name, param in model.named_parameters():
            param.requires_grad = False
            if tuning_mode == 'linear_probe':
                if "head." in name:
                    param.requires_grad = True
            elif tuning_mode == 'ssf':
                if "head." in name or "bases" in name or "ssf_scale" in name or "ssf_shift_"   in name:
                    param.requires_grad = True
            elif tuning_mode == 'dcf':

                if "head." in name or "bases" in name or "ssf_scale" in name or "ssf_shift_"  in name or "blocks.11."  in name:
                    param.requires_grad = True
            elif tuning_mode == 'attn_dcf_vo' or tuning_mode == 'attn_dcf_kv':
                if "head." in name or "ssf_scale" in name or "ssf_shift_" in name:
                    param.requires_grad = True
            elif tuning_mode == 'ssf_cnn':
                if "head." in name or "sc_scale" in name:
                    param.requires_grad = True
            elif tuning_mode == 'full':

                for name, param in model.named_parameters():
                    param.requires_grad = True
            elif tuning_mode == 'freeze_attn_mlp':
                for name, param in model.named_parameters():

                    param.requires_grad = True

                    if 'attn.' in name or 'mlp.' in name:
                        param.requires_grad = False

            elif tuning_mode == 'freeze_attn':
                for name, param in model.named_parameters():
                    param.requires_grad = True

                    if '.attn.' in name:
                        param.requires_grad = False

                    elif '.mlp.' in name:

                        try:
                            mlp_idx = name.find('.mlp.')
                            prefix = name[:mlp_idx]
                            parts = prefix.split('.')
                            layer_index = None
                            for part in reversed(parts):
                                if part.isdigit():
                                    layer_index = int(part)
                                    break
                            if layer_index is not None and layer_index < 6:
                                param.requires_grad = False
                        except Exception:
                            pass
            else:
                print(f"Warning: Unknown tuning_mode '{tuning_mode}'. Freezing all parameters or consider adding logic.")

        print('Parameter freezing based on tuning_mode finished!')
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_trainable_params}")
    # --------------------------------------------------------

    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay

    parameters = model.parameters() # 默认
    if hasattr(args, 'tuning_mode') and args.tuning_mode:

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
    elif weight_decay and filter_bias_and_bn:
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
        parameters = model.parameters()

    parameters_for_optimizer = parameters
    if hasattr(args, 'tuning_mode') and args.tuning_mode == 'attn_dcf':

        if not isinstance(parameters, list):
            bases_lr = getattr(args, 'bases_lr', args.lr)
            bases_decay = getattr(args, 'bases_decay', 0.0)

            parameters_groups = [
                {'params': [p for n, p in model.named_parameters() if 'bases' in n and p.requires_grad],
                 'lr': bases_lr, 'weight_decay': bases_decay},
                {'params': [p for n, p in model.named_parameters() if 'bases' not in n and 'bias' not in n and p.requires_grad],
                 'lr': args.lr, 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.named_parameters() if 'bias' in n and p.requires_grad],
                 'lr': args.lr, 'weight_decay': 0.},
            ]
            parameters_for_optimizer = parameters_groups
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
            try:
                optimizer = optim.Nadam(parameters_for_optimizer, **opt_args)
            except AttributeError:
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
            optimizer = Lookahead(optimizer)

    return optimizer
