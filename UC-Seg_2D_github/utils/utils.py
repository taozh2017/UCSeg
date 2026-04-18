import torch
from medpy import metric


def val_coef_ACDC(y_true, y_pred, classes, thr=0.5, epsilon=0.001):
    dice_list = []
    for c in range(1, classes):
        y_true_c = y_true.to(torch.float32)[:, c, ...].cpu().detach().numpy()
        y_pred_c = (y_pred > thr).to(torch.float32)[:, c, ...].cpu().detach().numpy()
        inter_map = y_true_c * y_pred_c
        inter = inter_map.sum()
        den = y_true_c.sum() + y_pred_c.sum()
        dice = ((2 * inter) / (den + epsilon)) if den > 0 else 0
        dice_list.append(dice)
    dice = sum(dice_list) / len(dice_list)
    return dice

def dice_coef(y_true, y_pred, thr=0.5, epsilon=0.001):
    y_true = y_true.to(torch.float32).squeeze(0)[1].cpu().detach().numpy()
    y_pred = (y_pred > thr).to(torch.float32).squeeze(0)[1].cpu().detach().numpy()
    inter_map = y_true * y_pred
    inter = inter_map.sum()
    den = y_true.sum() + y_pred.sum()
    dice = ((2 * inter) / (den + epsilon)) if den > 0 else 0
    return dice


def eval(y_true, y_pred, thr=0.5):
    y_true = y_true.to(torch.float32).squeeze(0)[1].cpu().detach().numpy()
    y_pred = (y_pred > thr).to(torch.float32).squeeze(0)[1].cpu().detach().numpy()
    
    
    single_class_res = []
    single_class_res.append(metric.binary.dc(y_pred, y_true))
    single_class_res.append(metric.binary.jc(y_pred, y_true))

    
    return single_class_res



def val_coef(y_true, y_pred, thr=0.5, epsilon=0.001):
    y_true = y_true.to(torch.float32)[:, 1, ...].cpu().detach().numpy()
    y_pred = (y_pred > thr).to(torch.float32)[:, 1, ...].cpu().detach().numpy()
    inter_map = y_true * y_pred
    inter = inter_map.sum()
    den = y_true.sum() + y_pred.sum()
    dice = ((2 * inter) / (den + epsilon)) if den > 0 else 0

    return dice



def patients_to_slices(dataset, patiens_num):
    ref_dict = {}
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "tumor" in dataset:
        ref_dict = {"10": 145, "30": 435, }
    elif "ISIC" in dataset:
        ref_dict = {"10": 207, "30": 622, }
    elif "thyroid" in dataset:
        ref_dict = {"10": 613, "30": 1841, }
    elif "BrainMRI" in dataset:
        ref_dict = {"10": 103, "30": 310, }
    elif "MRI_Hippocampus_Seg" in dataset:
        ref_dict = {"10": 282, "30": 846, }
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


