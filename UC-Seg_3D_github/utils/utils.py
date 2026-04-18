import torch
from medpy import metric


def calculate_metric_percase(y_true, y_pred, thr=0.5):
    y_true = y_true[1, ...].to(torch.float32).cpu().detach().numpy().astype(bool)
    y_pred = (y_pred > thr).astype(bool)

    if y_pred.sum() == 0:
        dice = 0
        jc = 0
        hd = 100
        asd = 100
    else:
        dice = metric.binary.dc(y_pred, y_true)
        jc = metric.binary.jc(y_pred, y_true)
        hd = metric.binary.hd95(y_pred, y_true)
        asd = metric.binary.asd(y_pred, y_true)
    return dice, jc, hd, asd


def val_coef(y_true, y_pred, thr=0.5, epsilon=0.001):
    y_true = y_true.to(torch.float32)[:, 1, ...].cpu().detach().numpy()
    y_pred = (y_pred > thr).to(torch.float32)[:, 1, ...].cpu().detach().numpy()
    inter_map = y_true * y_pred
    inter = inter_map.sum()
    den = y_true.sum() + y_pred.sum()
    # dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    dice = ((2 * inter) / (den + epsilon)) if den > 0 else 0
    # return dice, y_true, y_pred
    return dice



def patients_to_slices(dataset, patiens_num):
    ref_dict = {}
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "tumor" in dataset:
        ref_dict = {"30": 435}
    elif "ISIC" in dataset:
        ref_dict = {"30": 622}
    elif "LA_Seg_Training" in dataset:
        ref_dict = {"5": 4,"10": 8}
    elif "Pancreas" in dataset:
        ref_dict = {"5": 3, "10": 6, "20": 12, "30": 18}

    elif "BraTS" in dataset:
        ref_dict = {"5": 12,"10": 25}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]



def one_hot_encoder(input_tensor, n_classes):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()
