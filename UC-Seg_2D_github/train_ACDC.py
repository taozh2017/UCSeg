import argparse
import numpy as np
import random
import torch
import os
import logging

from UC_trainer_ACDC import UC_trainer
from tqdm import tqdm
from dataloader.dataset import build_Dataset
from torch.utils.data import DataLoader
from utils.utils import patients_to_slices
from utils.transforms import build_transforms

from dataloader.TwoStreamBatchSampler import TwoStreamBatchSampler

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data_2D',
                    help='Name of Experiment')

parser.add_argument('--dataset', type=str, default='/ACDC',
                    help='Name of Experiment')

parser.add_argument('--labeled_num', type=int, default=7,
                    help='Percentage of label quantity')

parser.add_argument('--unsup_factor', type=int, default=150,
                    help='unsup_factor')
parser.add_argument('--embed_c_factor', type=int, default=0.2,
                    help='embed_c_factor ')
parser.add_argument('--embed_p_factor', type=int, default=0.1,
                    help='embed_p_factor')

parser.add_argument('--nms', type=int,  default=1,
                    help='output channel of network')

parser.add_argument('--patch_size', type=list,  default=16,
                    help='patch size of network input')

parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--in_channels', type=int,  default=3,
                    help='in_channels')
parser.add_argument('--img_size', type=list,  default=[256, 256],
                    help='patch size of network input')

parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--seed', type=int,  default=42,
                    help='random seed')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--max_iterations', type=int, default=50000,
                    help='maximum epoch number to train')

parser.add_argument('--n_fold', type=int, default=1,
                    help='maximum epoch number to train')
parser.add_argument('--consistency', type=float, default=0.1,
                    help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

args = parser.parse_args()
# config = get_config(args)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def train(args, snapshot_path, logger):
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    
    # model
    trainer = UC_trainer(args)
    # dataset
    data_transforms = build_transforms(args)
    if args.dataset == "/ACDC":
        train_dataset = build_Dataset(data_dir=args.data_path + args.dataset, split="train_acdc_list",
                                      transform=data_transforms["train"])
        val_dataset = build_Dataset(data_dir=args.data_path + args.dataset, split="val_acdc_list",
                                    transform=data_transforms["valid_test"])
    else:
        train_dataset = build_Dataset(data_dir=args.data_path + args.dataset, split="train_semi",
                                      transform=data_transforms["train"])
        val_dataset = build_Dataset(data_dir=args.data_path + args.dataset, split="val",
                                    transform=data_transforms["valid_test"])

    # sampler
    total_slices = len(train_dataset)
    labeled_slice = patients_to_slices(args.dataset, args.labeled_num)
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler,
                             num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    logger.info("{} iterations per epoch".format(len(train_loader)))
    max_epoch = max_iterations // len(train_loader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    iter_num = 0
    for _ in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            trainer.train_(volume_batch, label_batch, iter_num, logger)
            iter_num = iter_num + 1
            if iter_num > 0 and iter_num % 200 == 0:
                trainer.val(val_loader, snapshot_path, iter_num, logger)


if __name__ == '__main__':
    import shutil
    for fold in range(args.n_fold):
        random.seed(fold*50 + 42)
        np.random.seed(fold*100 + 42)
        torch.manual_seed(fold*1000 + 42)
        torch.cuda.manual_seed(fold*10000 + 42)

        snapshot_path = "./Result/result_ACDC_10/fold_" + str(fold)

        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
        if not os.path.exists(snapshot_path + '/code'):
            os.makedirs(snapshot_path + '/code')

        logger = logging.getLogger('my_logger')
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(snapshot_path, "log.txt"))
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("Log file created successfully.")
        
        train(args, snapshot_path, logger)