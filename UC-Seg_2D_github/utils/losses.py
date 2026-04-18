import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
CE = torch.nn.BCELoss()
mse = torch.nn.MSELoss()


def loss_diff1(u_prediction_1, u_prediction_2):
    loss_a = 0.0

    for i in range(u_prediction_2.size(1)):
        loss_a = CE(u_prediction_1[:, i, ...].clamp(1e-8, 1 - 1e-7),
                                 Variable(u_prediction_2[:, i, ...].float(), requires_grad=False))

    loss_diff_avg = loss_a.mean().item()
    return loss_diff_avg


def loss_diff2(u_prediction_1, u_prediction_2):
    loss_b = 0.0

    for i in range(u_prediction_2.size(1)):
        loss_b = CE(u_prediction_2[:, i, ...].clamp(1e-8, 1 - 1e-7),
                                 Variable(u_prediction_1[:, i, ...], requires_grad=False))

    loss_diff_avg = loss_b.mean().item()
    return loss_diff_avg


def dice_loss(pred, label, epsilon=1e-5):
    intersection = torch.sum(pred * label, dim=(2, 3))
    union = torch.sum(pred, dim=(2, 3)) + torch.sum(label, dim=(2, 3))
    dice_coefficient = (2.0 * intersection + epsilon) / (union + epsilon)
    dice_loss = 1.0 - dice_coefficient
    return dice_loss.mean()



def mse_loss(input1, input2):
    return torch.mean((input1 - input2) ** 2)



