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

    reg_loss = 0.0
    count = 0
    for name, param in model.named_parameters():
        if 'bases_coeff' in name and param.requires_grad:
            W = param
            h = W.size(0)  # num_heads
            I = torch.eye(h, device=W.device)
            T = W + I

            if reg_type == 'ortho':

                ortho_loss = torch.norm(T.T @ T - I, p='fro') ** 2
                reg_loss += ortho_loss
            elif reg_type == 'rank':

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

    current_batch_size = query_feat.shape[0]
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
    sim_mat = sim_mat.to(dtype=score.dtype, device=score.device)
    
    new_logits = score @ sim_mat  # now both are float16 or float32
    new_logits = new_logits / temperature

    loss = F.cross_entropy(new_logits, label, reduction='none')
    return (loss * mask).mean()


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
    prob_1 = prob_1.float()
    prob_2 = prob_2.float()
    sim_mat = sim_mat.float()

    b1 = prob_1.shape[0]
    b2 = prob_2.shape[0]
    cls_num = prob_1.shape[1]

    if expand:
        prob_1 = prob_1.unsqueeze(2).unsqueeze(1).expand(-1, b2, -1, -1)  # B1xB2xCx1
        prob_2 = prob_2.unsqueeze(1).unsqueeze(0).expand(b1, -1, -1, -1)  # B1xB2x1xC

        prob_1 = prob_1.reshape(b1 * b2, cls_num, 1)
        prob_2 = prob_2.reshape(b1 * b2, 1, cls_num)

        sim = torch.sum(torch.bmm(prob_1, prob_2) * sim_mat, dim=(-1, -2))
        sim = sim.reshape(b1, b2)
    else:
        prob_1 = prob_1.unsqueeze(2)  # BxCx1
        prob_2 = prob_2.unsqueeze(1)  # Bx1xC

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

            fillvalue=(None, None)
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

        
        if is_binary:
            targets = targets.float().unsqueeze(-1)
        else:
            targets = targets.to(dtype=torch.int64)
            

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion, input_chans)

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
                

        if extra_samples is not None:
            extra_samples = extra_samples.float().to(device, non_blocking=True) / 100
            extra_samples = rearrange(extra_samples, 'B N (A T) -> B N A T', T=200)
        
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    teacher = model_ema.ema if model_ema is not None else model
                    output1 = test_class_batch(teacher, extra_samples, input_chans)
                    probs = F.softmax(output1, dim=-1)
                    max_probs, tgt_u = probs.max(dim=-1)
  

                    sim_mat = obtain_sim_mat(teacher)

            
            
            with torch.cuda.amp.autocast():
                mixed_samples = mixup_data_with_labels(extra_samples, tgt_u, samples, targets)
                output2 = test_class_batch(
                    model, mixed_samples, input_chans)
                    

            mask = max_probs.ge(0.8).float().detach()

            num_masked = torch.sum(mask)

            loss_1 = class_contrastive_loss(
                        score=output2,
                        label=tgt_u,
                        mask=mask,
                        sim_mat=sim_mat,
                        temperature=0.1
            )

            loss = loss + loss_1
            # neg_pool = torch.cat([output1, output2], dim=0)  # (3*B, C)
            # loss3 = instance_contrastive_loss(output1, output2, model, neg_pool)
            # loss = loss + loss3


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

        

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from einops import rearrange
import torch


@torch.no_grad()
def visualize_tsne(model, data_loader, device, ch_names=None, output_dir="tsne_plots", epoch=None, use_pca=False):

    model.eval()
    input_chans = utils.get_input_chans(ch_names) if ch_names is not None else None

    all_features = []
    all_labels = []

    for batch in data_loader:
        EEG_raw, target = batch[0], batch[1]  # [B, N, A*T]
        B, N, AT = EEG_raw.shape
        T = 200
        assert AT % T == 0, f"Expected time dim divisible by {T}, got {AT}"
        A = AT // T

        EEG = EEG_raw.float().to(device, non_blocking=True) / 100
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=T)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            features = model.forward_features(EEG, input_chans=input_chans)  # [B, D]

        all_features.append(features.cpu().numpy())
        all_labels.append(target.cpu().numpy())

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

    plt.figure(figsize=(8, 6))
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10.colors

    for i, label in enumerate(unique_labels):
        idx = (labels == label)
        plt.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            c=[colors[i % len(colors)]],
            s=25,
            alpha=0.5
        )

    plt.axis('off') 

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
