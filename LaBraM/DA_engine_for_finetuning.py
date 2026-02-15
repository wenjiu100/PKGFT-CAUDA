# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
import math
import sys
from typing import Iterable, Optional
import torch
from timm.utils import ModelEma
import utils
from einops import rearrange
import numpy as np
import torch.nn.functional as F
from itertools import zip_longest
from util import get_rank
import torch.nn as nn
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os




def compute_bases_coeff_regularization(model, reg_type, lambda_reg=1e-4):
    """
    计算 bases_coeff 的正则化损失
    """
    reg_loss = 0.0
    count = 0
    for name, param in model.named_parameters():
        if 'bases_coeff' in name and param.requires_grad:
            W = param
            h = W.size(0)  # num_heads
            I = torch.eye(h, device=W.device)
            T = W + I  # 实际使用的变换矩阵: bases_coeff + identity

            if reg_type == 'ortho':
                # 正交正则化：鼓励 T 是正交矩阵 (T^T @ T ≈ I)
                ortho_loss = torch.norm(T.T @ T - I, p='fro') ** 2
                reg_loss += ortho_loss
            elif reg_type == 'rank':
                # 低秩正则化：核范数（sum of singular values），鼓励 W 是低秩的
                U, S, V = torch.svd(W)
                rank_loss = torch.sum(S)  # nuclear norm
                reg_loss += rank_loss
            count += 1

    return lambda_reg * reg_loss if count > 0 else 0.0

def train_class_batch(model, samples, target, criterion, ch_names):
    outputs = model(samples, ch_names)
    loss = criterion(outputs, target)
    return loss, outputs

def instance_contrastive_loss(query_feat, key_feat, model, neg_feat):
    instance_contrastive_simmat = obtain_sim_mat(model).float()
    pos_logits = my_sim_compute(query_feat.float(), key_feat.float(), instance_contrastive_simmat, expand=False) * 0.5
    neg_logits = my_sim_compute(query_feat.float(), neg_feat.float(), instance_contrastive_simmat, expand=True) * 0.5
    all_logits = torch.cat((pos_logits, neg_logits), dim=1)
    #
    constrastive_labels = get_contrastive_labels(query_feat)
    info_nce_loss = F.cross_entropy(all_logits, constrastive_labels) * 0.7
    return info_nce_loss

def get_contrastive_labels(query_feat, device='cuda'):
    """
    为对比学习任务生成每个样本的唯一标识符。

    参数:
    - query_feat: 输入特征张量，形状为 (batch_size, ...)
    - device: 指定使用的设备，默认为 'cuda'，也可以是 'cpu'

    返回:
    - contrastive_labels: 形状为 (batch_size,) 的张量，包含从 0 到 batch_size-1 的整数
    """
    current_batch_size = query_feat.shape[0]
    # 生成从 0 到 current_batch_size - 1 的整数序列
    contrastive_labels = torch.arange(current_batch_size, dtype=torch.long, device=device)
    return contrastive_labels


def test_class_batch(model, samples, ch_names):
    outputs = model(samples, ch_names)

    return  outputs

def mixup_data(samples, alpha=0.1):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = samples.size()[0]
    index = torch.randperm(batch_size)

    mixed_samples = lam * samples + (1 - lam) * samples[index, :]
    return mixed_samples

import torch
import numpy as np

def stmixup_data(samples1, samples2):
    batch_size1 = samples1.size()[0]
    batch_size2 = samples2.size()[0]

    mixed_samples = []

    # Create a list of indices for samples2 to enable deletion of used elements.
    available_indices = list(range(batch_size2))

    for i in range(batch_size1):
        if available_indices:  # Check if there are still available samples in samples2.
            random_index = np.random.choice(available_indices)
            lam = np.random.beta(1, 1)  # Random mixing parameter from a Beta distribution.
            mixed_sample = lam * samples1[i] + (1 - lam) * samples2[random_index]
            mixed_samples.append(mixed_sample)
            del available_indices[available_indices.index(random_index)]  # Remove the used index.
        else:
            # If no more samples are available in samples2, stop mixing.
            break

    # Convert lists back to tensors and return.
    mixed_samples = torch.stack(mixed_samples)
    return mixed_samples

def mixup_data_with_labels(samples1, labels1, samples2, labels2):
    batch_size1 = samples1.size()[0]
    batch_size2 = samples2.size()[0]

    mixed_samples = []

    # Create a list of indices for samples2 to enable deletion of used elements.
    available_indices = list(range(batch_size2))

    for i in range(batch_size1):
        found_match = False
        for j in range(len(available_indices)):
            index_j = available_indices[j]
            if labels1[i].item() == labels2[index_j].item():
                # Perform Mixup with the matching sample from samples2 and remove it from available_indices.
                lam = 0.5  # Same label mixing parameter.
                mixed_sample = lam * samples1[i] + (1 - lam) * samples2[index_j]
                mixed_samples.append(mixed_sample)
                del available_indices[j]  # Remove the used index.
                found_match = True
                break

        if not found_match:
            # If no match was found, choose a random sample from the remaining samples2.
            if available_indices:  # Check if there are still available samples in samples2.
                random_index = np.random.choice(available_indices)
                lam = 0.9  # Different label mixing parameter.
                mixed_sample = lam * samples1[i] + (1 - lam) * samples2[random_index]
                mixed_samples.append(mixed_sample)
                del available_indices[available_indices.index(random_index)]  # Remove the used index.

    # Convert lists back to tensors and return along with original labels.
    mixed_samples = torch.stack(mixed_samples)
    # We keep the original labels from samples1 without modification.
    return mixed_samples


def apply_channel_mask(samples, mask_ratio=0.5):
    B, N, A, T = samples.size()
    num_channels_to_mask = int(N * mask_ratio)
    channels_to_mask = torch.randperm(N, device=samples.device)[:num_channels_to_mask]
    mask = torch.zeros(N, dtype=torch.bool, device=samples.device)
    mask[channels_to_mask] = True
    samples[:, mask, :, :] = 0
    return samples

def baseline_loss(online_output):

    loss_ent, loss_div = IM_loss(online_output)
    # batch_metrics['loss']['ent'] = loss_ent.item()
    # batch_metrics['loss']['div'] = loss_div.item()

    return loss_ent  + loss_div

def IM_loss(online_output):
    softmax_out = online_output
    loss_ent = -torch.mean(torch.sum(softmax_out * torch.log(softmax_out + 1e-5), 1)) * 0.5
    msoftmax = softmax_out.mean(dim=0)
    loss_div = torch.sum(msoftmax * torch.log(msoftmax + 1e-5)) * 0.5
    return loss_ent, loss_div



def class_contrastive_loss(score, label, mask, sim_mat, temperature=1.5):
    # ✅ 关键：让 sim_mat 与 score 同 dtype
    sim_mat = sim_mat.to(dtype=score.dtype, device=score.device)
    
    new_logits = score @ sim_mat  # now both are float16 or float32
    new_logits = new_logits / temperature

    loss = F.cross_entropy(new_logits, label, reduction='none')
    return (loss * mask).mean()

# def obtain_sim_mat(model, usage):
#     # 直接从 model 中获取权重
#     fc_weight = model.head.weight.detach()
#     normalized_fc_weight = F.normalize(fc_weight)
#     sim_mat_orig = normalized_fc_weight @ normalized_fc_weight.T

#     # 创建单位矩阵和非对角矩阵
#     num_classes = 6 # 假设有5个类别
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     eye_mat = torch.eye(num_classes).to(device).float()
#     non_eye_mat = 1 - eye_mat

#     # 应用非对角线缩放因子并组合
#     non_diag_alpha = 1.0
#     sim_mat = (eye_mat + non_eye_mat * sim_mat_orig * non_diag_alpha).clone()
#     return sim_mat
def obtain_sim_mat(model):
    fc_weight = model.head.weight.detach()  # 通常是 float32
    normalized = F.normalize(fc_weight, dim=1)
    sim_mat = normalized @ normalized.t()
    C = sim_mat.size(0)
    eye = torch.eye(C, device=sim_mat.device, dtype=sim_mat.dtype)
    sim_mat = eye + (1 - eye) * sim_mat
    return sim_mat  # 保持原始 dtype（float32）

def my_sim_compute(prob_1, prob_2, sim_mat, expand=True):
    """
    prob_1: B1xC
    prob_2: B2xC
    sim_mat: CxC
    expand: True, computation conducted between every element in prob_2 and prob_1; False, need B1=B2
    """
    # 确保所有输入张量是 float32 类型
    prob_1 = prob_1.float()
    prob_2 = prob_2.float()
    sim_mat = sim_mat.float()

    b1 = prob_1.shape[0]
    b2 = prob_2.shape[0]
    cls_num = prob_1.shape[1]

    if expand:
        prob_1 = prob_1.unsqueeze(2).unsqueeze(1).expand(-1, b2, -1, -1)  # B1xB2xCx1
        prob_2 = prob_2.unsqueeze(1).unsqueeze(0).expand(b1, -1, -1, -1)  # B1xB2x1xC

        # 在进行 bmm 操作之前，确保所有张量仍然是 float32 类型
        prob_1 = prob_1.reshape(b1 * b2, cls_num, 1)
        prob_2 = prob_2.reshape(b1 * b2, 1, cls_num)

        # 确保 sim_mat 也是 float32 类型
        sim = torch.sum(torch.bmm(prob_1, prob_2) * sim_mat, dim=(-1, -2))
        sim = sim.reshape(b1, b2)
    else:
        prob_1 = prob_1.unsqueeze(2)  # BxCx1
        prob_2 = prob_2.unsqueeze(1)  # Bx1xC

        # 确保 sim_mat 也是 float32 类型
        sim = torch.sum(torch.bmm(prob_1, prob_2) * sim_mat, dim=(-1, -2))
        sim = sim.reshape(b1, 1)

    return sim

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable,  test_data, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None, is_binary=True, bases_coeff_reg=None, reg_lambda=0.05,
                   ):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)

    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()



    for data_iter_step, ((samples, targets), (extra_samples, extra_targets)) in enumerate(zip_longest(
            metric_logger.log_every(data_loader, print_freq, header),
            metric_logger.log_every(test_data, print_freq, header),
            # metric_logger.log_every(val_data, print_freq, header),

            fillvalue=(None, None)  # 设置默认值为 None
    )):

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.float().to(device, non_blocking=True) / 100
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)



        targets = targets.to(device, non_blocking=True)
        # extra_targets = extra_targets.to(device, non_blocking=True)
        
        if is_binary:
            targets = targets.float().unsqueeze(-1)
            # extra_targets = extra_targets.float().unsqueeze(-1)
        else:
            targets = targets.to(dtype=torch.int64)  # 确保非二进制分类的目标是 int64 类型
            # extra_targets = extra_targets.to(dtype=torch.int64)
            

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion, input_chans)
                   # 添加正则化损失（仅在启用时）
            if bases_coeff_reg is not None:
                reg_loss = compute_bases_coeff_regularization(
                    model, 
                    reg_type=bases_coeff_reg, 
                    lambda_reg=reg_lambda
                )
                loss = loss+reg_loss
      
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion, input_chans)
                if bases_coeff_reg is not None:
                    reg_loss = compute_bases_coeff_regularization(
                        model, 
                        reg_type=bases_coeff_reg, 
                        lambda_reg=reg_lambda
                    )
                    loss = loss+reg_loss
                

            # 如果 extra_samples 存在，则处理额外的数据集样本
        if extra_samples is not None:
            extra_samples = extra_samples.float().to(device, non_blocking=True) / 100
            extra_samples = rearrange(extra_samples, 'B N (A T) -> B N A T', T=200)
        
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    teacher = model_ema.ema if model_ema is not None else model
                    output1 = test_class_batch(teacher, extra_samples, input_chans)
                    probs = F.softmax(output1, dim=-1)
                    max_probs, tgt_u = probs.max(dim=-1)
  

                     # Get fixed similarity matrix from teacher
                    sim_mat = obtain_sim_mat(teacher)

                    # output1 = test_class_batch( 
                    #     model, extra_samples, input_chans)
                
                # mixed_samples = mixup_data(extra_samples, alpha=0.1)

            # prob_threshold = 0.9

            # probs = torch.softmax(output1, dim=-1)          # 归一化为概率分布，每行和为 1
            # max_probs, tgt_u = torch.max(probs, dim=-1)     # max_probs ∈ [0, 1], tgt_u ∈ {0,1,2,3,4,5}


            
            
            with torch.cuda.amp.autocast():
                mixed_samples = mixup_data_with_labels(extra_samples, tgt_u, samples, targets)
                # mixed_samples = mixup_data(extra_samples, alpha=0.5)
              #   with torch.no_grad():
                output2 = test_class_batch(
                    model, mixed_samples, input_chans)
                    

            mask = max_probs.ge(0.8).float().detach()

            num_masked = torch.sum(mask)
            #print(f"Max prob stats: mean={max_probs.mean().item():.3f}, max={max_probs.max().item():.3f}")

            # loss_1 = class_contrastive_loss(output2, tgt_u, mask, model)

            # loss_1 = class_contrastive_loss(
            #             score=output2,
            #             label=tgt_u,
            #             mask=mask,
            #             sim_mat=sim_mat,
            #             temperature=0.1 # 可调
            # )
            # print("num_masked is {}".format(num_masked))

            # loss = loss + loss_1
            neg_pool = torch.cat([output1, output2], dim=0)  # (3*B, C)
            loss3 = instance_contrastive_loss(output1, output2, model, neg_pool)
            loss = loss + loss3


        loss_value = loss.item()

        # print(f"Main loss: {loss.item():.4f}, Contrastive loss: {loss_1.item():.4f}")

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if is_binary:
            class_acc = utils.get_metrics(torch.sigmoid(output).detach().cpu().numpy(), targets.detach().cpu().numpy(), ["accuracy"], is_binary)["accuracy"]
        else:
            class_acc = (output.max(-1)[-1] == targets.squeeze()).float().mean()

       
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        
        # 清理不再需要的变量
        # del samples, targets, extra_samples, extra_targets, output
        # torch.cuda.empty_cache()  # 清理未使用的 GPU 缓存


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



# --- 添加在 train.py 顶部的 import 区域（如果尚未导入）---
# --- t-SNE 可视化函数 ---
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from einops import rearrange
import torch

# @torch.no_grad()
# def visualize_tsne(model, data_loader, device, ch_names=None, output_dir="tsne_plots", epoch=None, use_pca=True):
#     """
#     可视化原始EEG输入和模型特征的t-SNE（纯散点图，无图注）。
    
#     Args:
#         model: 训练好的模型，需实现 forward_features(EEG, input_chans=...)
#         data_loader: DataLoader（建议用验证集）
#         device: 'cuda' or 'cpu'
#         ch_names: 通道名称列表（可选）
#         output_dir: 保存路径
#         epoch: 当前 epoch（用于命名）
#         use_pca: 是否先用PCA降到50维再t-SNE（推荐True）
#     """
#     model.eval()
#     input_chans = utils.get_input_chans(ch_names) if ch_names is not None else None

#     all_inputs_flat = []
#     all_features = []
#     all_labels = []

#     for batch in data_loader:
#         EEG_raw, target = batch[0], batch[1]  # [B, N, A*T]
#         B, N, AT = EEG_raw.shape
#         T = 200
#         assert AT % T == 0, f"Expected time dim divisible by {T}, got {AT}"
#         A = AT // T  # 动态推断 A，不再硬编码

#         # 保存原始输入（展平）
#         inputs_flat = EEG_raw.view(B, -1).cpu().numpy()  # [B, N*A*T]
#         all_inputs_flat.append(inputs_flat)

#         # 前向传播获取特征
#         EEG = EEG_raw.float().to(device, non_blocking=True) / 100
#         EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=T)
#         target = target.to(device, non_blocking=True)

#         with torch.cuda.amp.autocast():
#             features = model.forward_features(EEG, input_chans=input_chans)  # [B, D]

#         all_features.append(features.cpu().numpy())
#         all_labels.append(target.cpu().numpy())

#     # 合并所有批次
#     inputs_flat = np.concatenate(all_inputs_flat, axis=0)      # [N_total, D_in]
#     features = np.concatenate(all_features, axis=0)           # [N_total, D_feat]
#     labels = np.concatenate(all_labels, axis=0).squeeze()     # [N_total,]

#     os.makedirs(output_dir, exist_ok=True)

#     # === t-SNE on RAW INPUT ===
#     print("Running t-SNE on raw EEG input...")
#     if use_pca and inputs_flat.shape[1] > 50:
#         pca = PCA(n_components=50, random_state=42)
#         inputs_reduced = pca.fit_transform(inputs_flat)
#     else:
#         inputs_reduced = inputs_flat

#     # 标准化：对原始输入做 StandardScaler（关键！）
#     scaler = StandardScaler()
#     inputs_scaled = scaler.fit_transform(inputs_reduced)

#     tsne_input = TSNE(
#         n_components=2,
#         random_state=42,
#         init='pca',
#         learning_rate='auto',
#         perplexity=30,
#         n_iter=1000,
#         metric='euclidean'
#     ).fit_transform(inputs_scaled)
    
#     _plot_tsne_clean(tsne_input, labels, 
#                      save_path=os.path.join(output_dir, f"input_tsne_ep{epoch}.png" if epoch is not None else "input_tsne.png"))

#     # === t-SNE on MODEL FEATURES ===
#     print("Running t-SNE on model features...")
#     tsne_feat = TSNE(
#         n_components=2,
#         random_state=42,
#         init='pca',
#         learning_rate='auto',
#         perplexity=30,
#         n_iter=1000,
#         metric='euclidean'
#     ).fit_transform(features)
    
#     _plot_tsne_clean(tsne_feat, labels,
#                      save_path=os.path.join(output_dir, f"feature_tsne_ep{epoch}.png" if epoch is not None else "feature_tsne.png"))

@torch.no_grad()
def visualize_tsne(model, data_loader, device, ch_names=None, output_dir="tsne_plots", epoch=None, use_pca=False):
    """
    仅可视化模型特征的 t-SNE（纯散点图，无图注）。
    
    Args:
        model: 训练好的模型，需实现 forward_features(EEG, input_chans=...)
        data_loader: DataLoader（建议用验证集）
        device: 'cuda' or 'cpu'
        ch_names: 通道名称列表（可选）
        output_dir: 保存路径
        epoch: 当前 epoch（用于命名）
        use_pca: 是否先用PCA降到50维再t-SNE（对特征通常不需要，设为False）
    """
    model.eval()
    input_chans = utils.get_input_chans(ch_names) if ch_names is not None else None

    all_features = []
    all_labels = []

    for batch in data_loader:
        EEG_raw, target = batch[0], batch[1]  # [B, N, A*T]
        B, N, AT = EEG_raw.shape
        T = 200
        assert AT % T == 0, f"Expected time dim divisible by {T}, got {AT}"
        A = AT // T  # 动态推断 A，不再硬编码

        # 前向传播获取特征
        EEG = EEG_raw.float().to(device, non_blocking=True) / 100
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=T)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            features = model.forward_features(EEG, input_chans=input_chans)  # [B, D]

        all_features.append(features.cpu().numpy())
        all_labels.append(target.cpu().numpy())

    # 合并所有批次
    features = np.concatenate(all_features, axis=0)           # [N_total, D_feat]
    labels = np.concatenate(all_labels, axis=0).squeeze()     # [N_total,]

    os.makedirs(output_dir, exist_ok=True)

    # === t-SNE on MODEL FEATURES ONLY ===
    print("Running t-SNE on model features...")
    if use_pca and features.shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        features_reduced = pca.fit_transform(features)
    else:
        features_reduced = features

    tsne_feat = TSNE(
        n_components=2,
        random_state=42,
        init='pca',
        learning_rate='auto',
        perplexity=30,
        n_iter=1000,
        metric='euclidean'
    ).fit_transform(features_reduced)
    
    _plot_tsne_clean(tsne_feat, labels,
                     save_path=os.path.join(output_dir, f"feature_tsne_ep{epoch}.png" if epoch is not None else "feature_tsne.png"))
def _plot_tsne_clean(embedding, labels, save_path=None):
    """
    绘制纯散点图：无坐标轴、无图例、无标题，仅保留彩色散点。
    """
    plt.figure(figsize=(8, 6))
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10.colors  # 支持最多10类，循环使用

    for i, label in enumerate(unique_labels):
        idx = (labels == label)
        plt.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            c=[colors[i % len(colors)]],
            s=25,
            alpha=0.5
        )

    plt.axis('off')  # 完全隐藏坐标轴

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"✅ Saved clean t-SNE plot to: {save_path}")
    else:
        plt.show()
    plt.close()

@torch.no_grad()
def evaluate(data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'], is_binary=True):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #header = 'Test:'

    # switch to evaluation mode
    model.eval()
    pred = []
    true = []
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        EEG = batch[0]
        target = batch[1]
        EEG = EEG.float().to(device, non_blocking=True) / 100
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
        target = target.to(device, non_blocking=True)
        if is_binary:
            target = target.float().unsqueeze(-1)
        else:
            target = target.to(dtype=torch.int64)

        
        # compute output
        with torch.cuda.amp.autocast():
            output = model(EEG, input_chans=input_chans)
            loss = criterion(output, target)
        
        if is_binary:
            output = torch.sigmoid(output).cpu()
        else:
            output = output.cpu()
        target = target.cpu()

        # results = utils.get_metrics(output.numpy(), target.numpy(), metrics, is_binary)
        results = utils.get_metrics(output.detach().cpu().numpy(), target.cpu().numpy(), metrics, is_binary)
        pred.append(output)
        true.append(target)

        batch_size = EEG.shape[0]
        metric_logger.update(loss=loss.item())
        for key, value in results.items():
            metric_logger.meters[key].update(value, n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))
    
    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()
    np.set_printoptions(threshold=np.inf)
    pred_max_indices = np.argmax(pred, axis=1)

    # print('pred',pred_max_indices)
    # print('true',true)


    ret = utils.get_metrics(pred, true, metrics, is_binary, 0.5)
    ret['loss'] = metric_logger.loss.global_avg
    return ret
