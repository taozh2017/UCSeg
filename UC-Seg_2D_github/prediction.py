import torch
import argparse

from dataloader.dataset import build_Dataset
from utils.transforms import build_transforms
from torch.utils.data import DataLoader
import numpy as np
from utils.utils import eval


def get_entropy_map(p):
    ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
    return ent_map


if __name__ == '__main__':
    
    from module.UC_Seg_2D import UC_Seg

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='/opt/data/private/data_2D',
                        help='Name of Experiment')
    parser.add_argument('--dataset', type=str, default='/tumor_10',
                        help='Name of Experiment')
    # parser.add_argument('--dataset', type=str, default='/ISIC_TrainDataset_10',
    #                     help='Name of Experiment')
    # parser.add_argument('--dataset', type=str, default='/BrainMRI_10',
    #                     help='Name of Experiment')
    # parser.add_argument('--dataset', type=str, default='/thyroid_10',
    #                     help='Name of Experiment')
    # parser.add_argument('--dataset', type=str, default='/MRI_Hippocampus_Seg_30',
    #                     help='Name of Experiment')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='output channel of network')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='input channel of network')
    parser.add_argument('--img_size', type=list, default=[224, 224],
                        help='patch size of network input')
    parser.add_argument('--patch_size', type=list, default=14,
                        help='patch size of network input')

    parser.add_argument('--UC_model_path', type=str,
                        default="./best_model.pth",
                        help='model weight path')

    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    data_transforms = build_transforms(args)
    # test_dataset_list = ["test_Kvasir"]
    # test_dataset_list = ["test_CVC-ColonDB", ]
    test_dataset_list = ["test_CVC-300", "test_CVC-ClinicDB", "test_CVC-ColonDB", "test_ETIS-LaribPolypDB", "test_Kvasir"]
    # test_dataset_list = ["test_ISIC2018"]
    # test_dataset_list = ["test_DDTI", "test_tn3k"]
    # test_dataset_list = ["test_tn3k"]
    # test_dataset_list = ["test_BrainMRI"]
    for test_dataset_name in test_dataset_list:
        test_dataset = build_Dataset(data_dir=args.data_path + args.dataset, split=test_dataset_name,
                                     transform=data_transforms["valid_test"])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

        UC_model = UC_Seg(args).cuda()
        UC_checkpoint = torch.load(args.UC_model_path)
        UC_model.load_state_dict(UC_checkpoint)
        UC_model.eval()

        avg_dice_list = []
        avg_iou_list = []

        for i_batch, sampled_batch in enumerate(test_loader):
            test_image, test_label = sampled_batch["image"].cuda(), sampled_batch["label"].cuda()
            outputs_1,  embedding_1, outputs_2, embedding_2, pseudo_map = UC_model(test_image)

            pseudo_map_sig = torch.nn.Sigmoid()(pseudo_map)
            eval_list = eval(test_label, pseudo_map_sig, thr=0.5)

            avg_dice_list.append(eval_list[0])
            avg_iou_list.append(eval_list[1])

        avg_dice = np.mean(avg_dice_list)
        avg_iou = np.mean(avg_iou_list)

        
        print(test_dataset_name, " :")
        print("avg_dice: ", avg_dice)
        print("avg_iou: ", avg_iou)

