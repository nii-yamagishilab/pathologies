# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import math
import torch
import torch.nn.functional as F


def cross_entropy(logits, targets, **kwargs):
    ce_loss = F.cross_entropy(logits, targets)
    return ce_loss, ce_loss, None


def label_smoothing(logits, targets, **kwargs):
    ce_loss = F.cross_entropy(logits, targets)
    if "reduced_logits" in kwargs:
        logits = kwargs["reduced_logits"]
    log_p = F.log_softmax(logits, dim=-1)
    K = log_p.size(-1)
    H_u_p = (1 / K) * (-log_p.sum(dim=-1))
    reg_loss = kwargs["beta"] * H_u_p.mean()
    loss = ce_loss + reg_loss
    return loss, ce_loss, reg_loss


def confidence_penalty(logits, targets, **kwargs):
    ce_loss = F.cross_entropy(logits, targets)
    if "reduced_logits" in kwargs:
        logits = kwargs["reduced_logits"]
    log_p = F.log_softmax(logits, dim=-1)
    p = torch.exp(log_p)
    H_p = -p.mul(log_p).sum(dim=-1)
    reg_loss = kwargs["beta"] * H_p.mean()
    loss = ce_loss - reg_loss
    return loss, ce_loss, reg_loss


def j_penalty(logits, targets, **kwargs):
    ce_loss = F.cross_entropy(logits, targets)
    if "reduced_logits" in kwargs:
        logits = kwargs["reduced_logits"]
    log_p = F.log_softmax(logits, dim=-1)
    p = torch.exp(log_p)
    K = log_p.size(-1)
    H_u_p = (1.0 / K) * (-log_p.sum(dim=-1))
    H_p = -p.mul(log_p).sum(dim=-1)
    J = H_u_p - H_p
    reg_loss = kwargs["beta"] * J.mean()
    loss = ce_loss + reg_loss
    return loss, ce_loss, reg_loss


def js_penalty(logits, targets, **kwargs):
    ce_loss = F.cross_entropy(logits, targets)
    if "reduced_logits" in kwargs:
        logits = kwargs["reduced_logits"]
    log_p = F.log_softmax(logits, dim=-1)
    p = torch.exp(log_p)
    K = log_p.size(-1)
    H_p = -p.mul(log_p).sum(dim=-1)
    u = (1.0 / K) * torch.ones_like(p)
    H_u = math.log(K)
    m = 0.5 * (p + u)
    H_m = -m.mul(torch.log(m)).sum(dim=-1)
    JS = H_m - (0.5 * H_p) - (0.5 * H_u)
    JS = torch.clamp(JS, min=1e-10)
    reg_loss = kwargs["beta"] * JS.mean()
    loss = ce_loss + reg_loss
    return loss, ce_loss, reg_loss
