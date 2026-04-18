import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn


class DiceLoss(nn.Module):

    def __init__(self, alpha=0.5, beta=0.5, size_average=True, reduce=True):
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.size_average = size_average
        self.reduce = reduce

    def forward(self, preds, targets, weight=False):
        N = preds.size(0)
        C = preds.size(1)
        if preds.ndim==5:
            preds = preds.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        else:
            preds = preds.permute(0, 2, 3, 1).contiguous().view(-1, C)
        # targets = targets.view(-1, 1)
        targets = targets.view(-1, 1)

        log_P = F.log_softmax(preds, dim=1)
        P = torch.exp(log_P)
        # P = F.softmax(preds, dim=1)
        smooth = torch.zeros(C, dtype=torch.float32).fill_(0.00001)

        class_mask = torch.zeros(preds.shape).to(preds.device) + 1e-8
        class_mask.scatter_(1, targets, 1.)

        ones = torch.ones(preds.shape).to(preds.device)
        P_ = ones - P
        class_mask_ = ones - class_mask

        TP = P * class_mask
        FP = P * class_mask_
        FN = P_ * class_mask

        smooth = smooth.to(preds.device)
        self.alpha = FP.sum(dim=(0)) / ((FP.sum(dim=(0)) + FN.sum(dim=(0))) + smooth)

        self.alpha = torch.clamp(self.alpha, min=0.2, max=0.8)
        #print('alpha:', self.alpha)
        self.beta = 1 - self.alpha
        num = torch.sum(TP, dim=(0)).float()
        den = num + self.alpha * torch.sum(FP, dim=(0)).float() + self.beta * torch.sum(FN, dim=(0)).float()

        dice = num / (den + smooth)

        if not self.reduce:
            loss = torch.ones(C).to(dice.device) - dice
            return loss
        loss = 1 - dice
        if weight is not False:
            loss *= weight.squeeze(0)
        loss = loss.sum()
        if self.size_average:
            if weight is not False:
                loss /= weight.squeeze(0).sum()
            else:
                loss /= C
        return loss

def KL(alpha, c):
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    beta = torch.ones((1, c)).cuda()
    # Mbeta = torch.ones((alpha.shape[0],c)).cuda()
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def TDice(output, target, criterion_dl):
    dice = criterion_dl(output, target)
    return dice


def dce_evidence_u_loss(p, alpha, c, current_step, lamda_step, total_step, eps, disentangle, evidence):
    # c: class number
    p = torch.argmax(p, dim=1).unsqueeze(1)
    criterion_dl = DiceLoss()
    if alpha.ndim == 5:
        soft_p = p.unsqueeze(1)
    else:
        soft_p = p

    L_dice = TDice(evidence, soft_p, criterion_dl)

    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    label = label.view(-1, c)

    # digama loss
    L_ace = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    # KL loss
    annealing_coef = min(1, current_step / lamda_step)
    annealing_start = torch.tensor(0.01, dtype=torch.float32)
    annealing_AU = annealing_start * torch.exp(-torch.log(annealing_start) / total_step * current_step)
    alp = E * (1 - label) + 1
    L_KL = annealing_coef * KL(alp, c)

    # AU Loss
    pred_scores, pred_cls = torch.max(alpha / S, 1, keepdim=True)
    uncertainty = c / S
    # target = p.view(-1, 1)
    target = p.view(-1, 1)
    acc_match = torch.reshape(torch.eq(pred_cls, target).float(), (-1, 1))
    if disentangle:
        acc_uncertain = - torch.log(pred_scores * (1 - uncertainty) + eps)
        inacc_certain = - torch.log((1 - pred_scores) * uncertainty + eps)
    else:
        acc_uncertain = - pred_scores * torch.log(1 - uncertainty + eps)
        inacc_certain = - (1 - pred_scores) * torch.log(uncertainty + eps)
    L_AU = annealing_AU * acc_match * acc_uncertain + (1 - annealing_AU) * (1 - acc_match) * inacc_certain
    return (L_ace  + L_KL + (1 - annealing_AU)*L_dice + L_AU)


def u_loss(p, alpha, c, current_step, total_step, eps, disentangle):
    # c: class number
    p = torch.argmax(p, dim=1).unsqueeze(1)
    # p = p.unsqueeze(1)

    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))
    S = torch.sum(alpha, dim=1, keepdim=True)

    # # digama loss
    # label = F.one_hot(p, num_classes=c)
    # label = label.view(-1, c)
    # L_ace = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    # KL loss
    annealing_start = torch.tensor(0.01, dtype=torch.float32)
    annealing_AU = annealing_start * torch.exp(-torch.log(annealing_start) / total_step * current_step)

    # annealing_coef = min(1, current_step / lamda_step)
    # E = alpha - 1
    # alp = E * (1 - label) + 1
    # L_KL = annealing_coef * KL(alp, c)

    # AU Loss
    pred_scores, pred_cls = torch.max(alpha / S, 1, keepdim=True)
    uncertainty = c / S
    target = p.view(-1, 1)
    acc_match = torch.reshape(torch.eq(pred_cls, target).float(), (-1, 1))
    if disentangle:
        acc_uncertain = - torch.log(pred_scores * (1 - uncertainty) + eps)
        inacc_certain = - torch.log((1 - pred_scores) * uncertainty + eps)
    else:
        acc_uncertain = - pred_scores * torch.log(1 - uncertainty + eps)
        inacc_certain = - (1 - pred_scores) * torch.log(uncertainty + eps)
    L_AU = annealing_AU * acc_match * acc_uncertain + (1 - annealing_AU) * (1 - acc_match) * inacc_certain

    return L_AU


def infer(input):
    """
    :param input: modal data
    :return: evidence of modal data
    """
    evidence = F.softplus(input)
    return evidence


def trans_infer(input):
    """
    :param input: modal data
    :return: evidence of modal data
    """
    evidence = F.softplus(input)
    return evidence


def pseudo_infer(input):
    """
    :param input: modal data
    :return: evidence of modal data
    """
    evidence = F.softplus(input)
    return evidence
