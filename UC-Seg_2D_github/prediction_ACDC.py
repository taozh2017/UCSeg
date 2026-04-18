import torch
from medpy import metric
from scipy.ndimage import zoom
import numpy as np
from utils.mix_up import get_ACDC_masks

import torch
import argparse

import torch.nn.functional as F

from dataloader.dataset import build_Dataset
from torch.utils.data import DataLoader

from utils.transforms import build_transforms


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice_res = []
    if pred.sum() > 0:
        dice_res.append(metric.binary.dc(pred, gt))
    else:
        dice_res.append(0)
    return dice_res

def test_single_volume(args, image, label, UC_seg):
    classes = args.num_classes
    patch_size = args.img_size
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)

        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        input = input.repeat(1,3,1,1)
        with torch.no_grad():
            outputs_1, embedding_1, outputs_2, embedding_2, pseudo_map = UC_seg(input)
            pseudo_map_soft = torch.softmax(pseudo_map, dim=1)
            out_pseudo_map = torch.argmax(pseudo_map_soft, dim=1).squeeze(0).cpu().detach().numpy()
            pred_pseudo_map = zoom(out_pseudo_map, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred_pseudo_map

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list



def one_hot_encoder(input_tensor, n_classes):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


if __name__ == '__main__':


    from module.UC_Seg_2D import UC_Seg

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='/opt/data/private/data_2D',
                        help='Name of Experiment')
    parser.add_argument('--dataset', type=str, default='/ACDC',
                        help='Name of Experiment')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='output channel of network')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='input channel of network')
    parser.add_argument('--img_size', type=list, default=[256, 256],
                        help='patch size of network input')
    parser.add_argument('--patch_size', type=list, default=16,
                        help='patch_size')

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--model_path', type=str,
                        default="./best_model.pth",
                        help='model weight path')
    args = parser.parse_args()



    data_transforms = build_transforms(args)

    test_dataset = build_Dataset(data_dir=args.data_path + args.dataset, split="test_acdc_list",
                                 transform=data_transforms["valid_test"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    UC_seg = UC_Seg(args).train().cuda()
    checkpoint = torch.load(args.model_path)
    UC_seg.load_state_dict(checkpoint)
    UC_seg.eval()
    avg_dice_list = 0.0
    avg_iou_list = 0.0
    avg_hd95_list = 0.0
    avg_asd_list = 0.0
    classes = args.num_classes
    patch_size = args.img_size
    final_res = [0, 0, 0, 0]

    for i_batch, sampled_batch in enumerate(test_loader):
        test_image, test_label = sampled_batch["image"].cuda(), sampled_batch["label"].cuda()
        image, label = test_image.squeeze(0).cpu().detach().numpy(), test_label.squeeze(0).cpu().detach().numpy()

        test_label = F.interpolate(test_label, size=(256, 256), mode='bilinear', align_corners=False)

        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)

            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            input = input.repeat(1, 3, 1, 1)
            with torch.no_grad():
                outputs_1, embedding_1, outputs_2, embedding_2, pseudo_map = UC_seg(input)
                
                out_UC_cuda = get_ACDC_masks(pseudo_map, nms=0)
                out_UC = out_UC_cuda.squeeze(0).cpu().detach().numpy()

                pred_UC = zoom(out_UC, (x / patch_size[0], y / patch_size[1]), order=0)

                prediction[ind] = pred_UC

        metric_list = []
        for i in range(1, classes):
            disc_pred = prediction == i
            gt = label == i
            disc_pred[disc_pred > 0] = 1
            gt[gt > 0] = 1
            single_class_res = []
            if disc_pred.sum() > 0:
                single_class_res.append(metric.binary.dc(disc_pred, gt))
                single_class_res.append(metric.binary.jc(disc_pred, gt))
                single_class_res.append(metric.binary.asd(disc_pred, gt))
                single_class_res.append(metric.binary.hd95(disc_pred, gt))
            else:
                single_class_res = [0, 0, 0, 0]
            metric_list.append(single_class_res)

        metric_list = np.array(metric_list).astype("float32")
        metric_list = np.mean(metric_list, axis=0)

        print(metric_list)
        final_res += metric_list
    final_res = [x / len(test_loader) for x in final_res]
    print("avg_dice: ", final_res[0])
    print("avg_iou: ", final_res[1])
    print("avg_asd: ", final_res[2])
    print("avg_hd95: ", final_res[3])