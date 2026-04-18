import os
import logging

import torch
import random
import torch.nn as nn
import numpy as np
from prediction_ACDC import test_single_volume

from utils.mix_up import get_ACDC_masks, generate_mask
from utils.losses import dice_loss, loss_diff1, loss_diff2

from module.denseCL_SSIM_global_2D import Contrast_global_consistency
from utils.UMIS import infer, trans_infer, u_loss, pseudo_infer
from utils.utils import val_coef_ACDC
from module.UC_Seg_2D import UC_Seg

logger = logging.getLogger()
logger.info("This is a log message from the second file.")

# ce_loss = smp.losses.SoftBCEWithLogitsLoss()
ce_loss = torch.nn.CrossEntropyLoss()


def one_hot_encoder(input_tensor, n_classes):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


class UC_trainer(nn.Module):
    def __init__(self, args):
        super(UC_trainer, self).__init__()
        self.best_performance_dice = 0.0
        self.best_performance_95hd = 100.0
        self.args = args
        self.UC_seg = UC_Seg(args).train().cuda()
        self.Diff_aware_1 = Contrast_global_consistency(args, 256).cuda()
        self.Diff_aware_2 = Contrast_global_consistency(args, 128).cuda()
        self.Diff_aware_3 = Contrast_global_consistency(args, 64).cuda()
        self.eps = 1e-10
        self.disentangle = False
        self.optimizer = torch.optim.SGD(self.UC_seg.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.00001)

    def UMIS(self, args, label_batch, outputs_1, outputs_2, iter_num, eps=1e-10, disentangle=False):
        # UMIS res
        evidence_1 = infer(outputs_1)
        alpha_1 = evidence_1 + 1
        y = label_batch[:args.labeled_bs].clone()
        UMIS_loss_1 = u_loss(y.to(torch.int64), alpha_1, args.num_classes, iter_num, args.max_iterations, eps, disentangle)
        UMIS_loss_1 = torch.mean(UMIS_loss_1)

        # UMIS trans
        evidence_2 = trans_infer(outputs_2)
        alpha_2 = evidence_2 + 1
        UMIS_loss_2 = u_loss(y.to(torch.int64), alpha_2, args.num_classes, iter_num, args.max_iterations, eps, disentangle)
        UMIS_loss_2 = torch.mean(UMIS_loss_2)

        return UMIS_loss_1, UMIS_loss_2, alpha_1, alpha_2

    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.args.consistency * self.sigmoid_rampup(epoch, self.args.consistency_rampup)


    def train_(self, volume_batch, label_batch, iter_num, logger):
        # infer
        outputs_1, embedding_1, outputs_2, embedding_2, pseudo_map = self.UC_seg(volume_batch)

        # labeled loss
        outputs_labeled_1 = outputs_1[:self.args.labeled_bs]
        outputs_labeled_2 = outputs_2[:self.args.labeled_bs]

        outputs_labeled_soft_1 = torch.softmax(outputs_labeled_1, dim=1)
        outputs_labeled_soft_2 = torch.softmax(outputs_labeled_2, dim=1)

        # UMIS
        UMIS_loss_1, UMIS_loss_2, alpha_1, alpha_2 = self.UMIS(self.args, label_batch, outputs_labeled_1, outputs_labeled_2, iter_num)

        # normal supervised loss & UMIS_loss
        supervised_loss_1 = (ce_loss(outputs_labeled_1, label_batch[:self.args.labeled_bs]) + UMIS_loss_1 +
                               dice_loss(outputs_labeled_soft_1, label_batch[:self.args.labeled_bs]))
        supervised_loss_2 = (ce_loss(outputs_labeled_2, label_batch[:self.args.labeled_bs]) + UMIS_loss_2 +
                                 dice_loss(outputs_labeled_soft_2, label_batch[:self.args.labeled_bs]))

        # infer labeled pseudo & loss
        labeled_pseudo_map = pseudo_map[:self.args.labeled_bs]

        # pseudo UMIS loss
        evidence_pseudo_map = pseudo_infer(labeled_pseudo_map)
        alpha_pseudo_map = evidence_pseudo_map + 1
        UMIS_loss_pseudo_map = u_loss(label_batch[:self.args.labeled_bs].clone().to(torch.int64), alpha_pseudo_map, self.args.num_classes, iter_num, self.args.max_iterations, self.eps, self.disentangle)
        UMIS_loss_pseudo_map = torch.mean(UMIS_loss_pseudo_map)

        # pseudo supervised loss
        labeled_pseudo_map_soft = torch.softmax(labeled_pseudo_map, dim=1)
        loss_pseudo_labeled_ce = ce_loss(labeled_pseudo_map, label_batch[:self.args.labeled_bs])
        loss_pseudo_labeled_dice = dice_loss(labeled_pseudo_map_soft, label_batch[:self.args.labeled_bs])
        # pseudo total loss
        loss_pseudo_labeled = loss_pseudo_labeled_dice + loss_pseudo_labeled_ce + UMIS_loss_pseudo_map

        # infer unlabeled
        outputs_unlabeled_1 = outputs_1[self.args.labeled_bs:]
        outputs_unlabeled_2 = outputs_2[self.args.labeled_bs:]
        outputs_unlabeled_soft_1 = torch.softmax(outputs_unlabeled_1, dim=1)
        outputs_unlabeled_soft_2 = torch.softmax(outputs_unlabeled_2, dim=1)

        # unlabeled pseudo map
        unlabeled_pseudo_map = pseudo_map[self.args.labeled_bs:]
        if self.args.nms:
            unlabeled_pseudo_map = get_ACDC_masks(unlabeled_pseudo_map, nms=1)
            unlabeled_pseudo_map = one_hot_encoder(unlabeled_pseudo_map.unsqueeze(1), n_classes=self.args.num_classes)
        else:
            unlabeled_pseudo_map_soft = torch.softmax(unlabeled_pseudo_map, dim=1)
            unlabeled_pseudo_map = (unlabeled_pseudo_map_soft > 0.5).to(torch.float32).detach()

        # consistency labeled loss
        outputs_soft_1 = torch.cat((outputs_labeled_soft_1, outputs_unlabeled_soft_1), dim=0)
        outputs_soft_2 = torch.cat((outputs_labeled_soft_2, outputs_unlabeled_soft_2), dim=0)
        loss_diff_1 = loss_diff1(outputs_soft_1, outputs_soft_2)
        loss_diff_2 = loss_diff2(outputs_soft_1, outputs_soft_2)

        lr_ = self.args.base_lr * (1.0 - iter_num / self.args.max_iterations)
        consistency_weight = self.get_current_consistency_weight(iter_num // 150) * self.args.unsup_factor
        embed_c_factor = self.args.embed_c_factor
        embed_p_factor = self.args.embed_p_factor

        # contrast loss
        label = torch.cat((label_batch[:self.args.labeled_bs], unlabeled_pseudo_map), dim=0)
        if self.args.nms:
            outputs_mask_1 = get_ACDC_masks(outputs_1, nms=1)
            outputs_mask_2 = get_ACDC_masks(outputs_2, nms=1)
            outputs_mask_1 = one_hot_encoder(outputs_mask_1.unsqueeze(1), n_classes=self.args.num_classes)
            outputs_mask_2 = one_hot_encoder(outputs_mask_2.unsqueeze(1), n_classes=self.args.num_classes)
            patch_loss_1, loss_embed_c_res_1, loss_embed_c_trans_1 = self.Diff_aware_1("contrast", embedding_1[-1], embedding_2[-1], outputs_mask_1, outputs_mask_2, is_mixed_image=False)
            patch_loss_2, loss_embed_c_res_2, loss_embed_c_trans_2 = self.Diff_aware_2("contrast", embedding_1[-2], embedding_2[-2], outputs_mask_1, outputs_mask_2, is_mixed_image=False)
            patch_loss_3, loss_embed_c_res_3, loss_embed_c_trans_3 = self.Diff_aware_3("contrast", embedding_1[-3], embedding_2[-3], outputs_mask_1, outputs_mask_2, is_mixed_image=False)
        else:
            patch_loss_1, loss_embed_c_res_1, loss_embed_c_trans_1 = self.Diff_aware_1("contrast", embedding_1[-1], embedding_2[-1], outputs_soft_1, outputs_soft_2, is_mixed_image=False)
            patch_loss_2, loss_embed_c_res_2, loss_embed_c_trans_2 = self.Diff_aware_2("contrast", embedding_1[-2], embedding_2[-2], outputs_soft_1, outputs_soft_2, is_mixed_image=False)
            patch_loss_3, loss_embed_c_res_3, loss_embed_c_trans_3 = self.Diff_aware_3("contrast", embedding_1[-3], embedding_2[-3], outputs_soft_1, outputs_soft_2, is_mixed_image=False)
        patch_loss = (patch_loss_1 + patch_loss_2 + patch_loss_3) / 3
        loss_embed_c_res = (loss_embed_c_res_1 + loss_embed_c_res_2 + loss_embed_c_res_3) / 3
        loss_embed_c_trans = (loss_embed_c_trans_1  + loss_embed_c_trans_2 + loss_embed_c_trans_3) / 3

        # save_feature
        dice_value_labeled_1 = val_coef_ACDC(outputs_labeled_soft_1, label_batch[:self.args.labeled_bs], self.args.num_classes)
        dice_value_labeled_2 = val_coef_ACDC(outputs_labeled_soft_2, label_batch[:self.args.labeled_bs], self.args.num_classes)
        dice_value_labeled = (dice_value_labeled_1 + dice_value_labeled_2) / 2
        ssim_thr = 0.5 + 0.46 * (iter_num / self.args.max_iterations)
        if dice_value_labeled_1 > ssim_thr:
            self.Diff_aware_1("interpolation_save", feature_1=embedding_1[-1], label=label, save_type = "feature_1")
            self.Diff_aware_2("interpolation_save", feature_1=embedding_1[-2], label=label, save_type = "feature_1")
            self.Diff_aware_3("interpolation_save", feature_1=embedding_1[-3], label=label, save_type = "feature_1")
        if dice_value_labeled_2 > ssim_thr:
            self.Diff_aware_1("interpolation_save", feature_2=embedding_2[-1], label=label, save_type = "feature_2")
            self.Diff_aware_2("interpolation_save", feature_2=embedding_2[-2], label=label, save_type = "feature_2")
            self.Diff_aware_3("interpolation_save", feature_2=embedding_2[-3], label=label, save_type = "feature_2")

        # unsupervised loss
        unsupervised_loss_1 = (ce_loss(outputs_unlabeled_1, unlabeled_pseudo_map) + dice_loss(outputs_unlabeled_soft_1, unlabeled_pseudo_map))
        unsupervised_loss_2 = (ce_loss(outputs_unlabeled_2, unlabeled_pseudo_map) + dice_loss(outputs_unlabeled_soft_2, unlabeled_pseudo_map))

        if iter_num < 12000:
            loss_1 = (supervised_loss_1 + loss_diff_1 * 0.5 + consistency_weight * unsupervised_loss_1 + embed_c_factor * (loss_embed_c_res))
            loss_2 = (supervised_loss_2 + loss_diff_2 * 0.5 + consistency_weight * unsupervised_loss_2 + embed_c_factor * (loss_embed_c_trans))

            loss = (loss_1 + loss_2) * 0.5 + loss_pseudo_labeled + embed_p_factor * patch_loss

            logger.info('iteration %d : '
                         
                         '  dice_value_labeled: : %f'
                         
                         '  net_1_loss : %f'
                         '  net_2_loss : %f'
                         '  loss_pseudo_labeled : %f '
                         '  patch_loss: %f'
                         
                         '  net_1_c_loss: %f'
                         '  net_1_supervised_loss: %f'
                         '  net_1_consistency_loss : %f '
                         '  net_1_unsupervised_loss : %f '

                         '  net_2_c_loss: %f'
                         '  net_2_supervised_loss: %f'
                         '  net_2_consistency_loss : %f '
                         '  net_2_unsupervised_loss : %f '
                         '  lr_ : %f'
                         % (iter_num,

                            dice_value_labeled,

                            loss_1.item(),
                            loss_2.item(),
                            loss_pseudo_labeled,
                            embed_p_factor * patch_loss,

                            embed_c_factor * loss_embed_c_res,
                            supervised_loss_1,
                            loss_diff_1,
                            consistency_weight * unsupervised_loss_1,

                            embed_c_factor * loss_embed_c_trans,
                            supervised_loss_2,
                            loss_diff_2,
                            consistency_weight * unsupervised_loss_2,
                            lr_)
                         )
        else:
            labeled_volume_batch = volume_batch[:self.args.labeled_bs]
            unlabeled_volume_batch = volume_batch[self.args.labeled_bs:]
            labeled_label_batch = label_batch[:self.args.labeled_bs]
            img_mask, loss_mask = generate_mask(labeled_volume_batch)
            random_number = random.random()

            if random_number > 0.5:
                mix_image = labeled_volume_batch * img_mask + unlabeled_volume_batch * (1 - img_mask)
                gt_mix_image = labeled_label_batch * img_mask + unlabeled_pseudo_map * (1 - img_mask)
            else:
                mix_image = labeled_volume_batch * (1 - img_mask) + unlabeled_volume_batch * img_mask
                gt_mix_image = labeled_label_batch * (1 - img_mask) + unlabeled_pseudo_map * img_mask

            outputs_mixed_1, embedding_mixed_1, outputs_mixed_2, embedding_mixed_2, pseudo_map_mixed = self.UC_seg(mix_image)
            outputs_mixed_soft_1 = torch.softmax(outputs_mixed_1, dim=1)
            outputs_mixed_soft_2 = torch.softmax(outputs_mixed_2, dim=1)

            # mix super loss
            mix_loss_1 = (ce_loss(outputs_mixed_1, gt_mix_image) + dice_loss(outputs_mixed_soft_1, gt_mix_image))
            mix_loss_2 = (ce_loss(outputs_mixed_2, gt_mix_image) + dice_loss(outputs_mixed_soft_2, gt_mix_image))

            # mix cons loss
            consistency_loss_mix_1 = loss_diff1(outputs_mixed_soft_1, outputs_mixed_soft_2)
            consistency_loss_mix_2 = loss_diff2(outputs_mixed_soft_1, outputs_mixed_soft_2)

            # mix contrast loss
            if self.args.nms:
                outputs_mixed_mask_1 = get_ACDC_masks(outputs_mixed_1, nms=1)
                outputs_mixed_mask_2 = get_ACDC_masks(outputs_mixed_2, nms=1)
                outputs_mixed_mask_1 = one_hot_encoder(outputs_mixed_mask_1.unsqueeze(1), n_classes=self.args.num_classes)
                outputs_mixed_mask_2 = one_hot_encoder(outputs_mixed_mask_2.unsqueeze(1), n_classes=self.args.num_classes)
                patch_loss_mix_1, loss_embed_c_res_mixed_1, loss_embed_c_trans_mixed_1 = (self.Diff_aware_1("contrast", embedding_mixed_1[-1], embedding_mixed_2[-1], outputs_mixed_mask_1, outputs_mixed_mask_2, is_mixed_image=True))
                patch_loss_mix_2, loss_embed_c_res_mixed_2, loss_embed_c_trans_mixed_2 = (self.Diff_aware_2("contrast", embedding_mixed_1[-2], embedding_mixed_2[-2], outputs_mixed_mask_1, outputs_mixed_mask_2, is_mixed_image=True))
                patch_loss_mix_3, loss_embed_c_res_mixed_3, loss_embed_c_trans_mixed_3 = (self.Diff_aware_3("contrast", embedding_mixed_1[-3], embedding_mixed_2[-3], outputs_mixed_mask_1, outputs_mixed_mask_2, is_mixed_image=True))
            else:
                patch_loss_mix_1, loss_embed_c_res_mixed_1, loss_embed_c_trans_mixed_1 = (self.Diff_aware_1("contrast", embedding_mixed_1[-1], embedding_mixed_2[-1], outputs_mixed_soft_1, outputs_mixed_soft_2, is_mixed_image=True))
                patch_loss_mix_2, loss_embed_c_res_mixed_2, loss_embed_c_trans_mixed_2 = (self.Diff_aware_2("contrast", embedding_mixed_1[-2], embedding_mixed_2[-2], outputs_mixed_soft_1, outputs_mixed_soft_2, is_mixed_image=True))
                patch_loss_mix_3, loss_embed_c_res_mixed_3, loss_embed_c_trans_mixed_3 = (self.Diff_aware_3("contrast", embedding_mixed_1[-3], embedding_mixed_2[-3], outputs_mixed_soft_1, outputs_mixed_soft_2, is_mixed_image=True))
            patch_loss_mix = (patch_loss_mix_1 + patch_loss_mix_2 + patch_loss_mix_3) / 3
            loss_embed_c_res_mixed = (loss_embed_c_res_mixed_1 + loss_embed_c_res_mixed_2 + loss_embed_c_res_mixed_3) / 3
            loss_embed_c_trans_mixed = (loss_embed_c_trans_mixed_1 + loss_embed_c_trans_mixed_2 + loss_embed_c_trans_mixed_3) / 3

            loss_1 = (supervised_loss_1 + (loss_diff_1 + consistency_loss_mix_1) * 0.5 * 0.5 +
                      consistency_weight * (unsupervised_loss_1 + mix_loss_1) * 0.5 +
                      embed_c_factor * (loss_embed_c_res + loss_embed_c_res_mixed) * 0.5
                      )
            loss_2 = (supervised_loss_2 + (loss_diff_2 + consistency_loss_mix_2) * 0.5 * 0.5 +
                      consistency_weight * (unsupervised_loss_2 + mix_loss_2) * 0.5 +
                      embed_c_factor * (loss_embed_c_trans + loss_embed_c_trans_mixed) * 0.5
                      )
            loss = (loss_1 + loss_2) * 0.5 + loss_pseudo_labeled + embed_p_factor * (patch_loss_mix + patch_loss) * 0.5

            logger.info('iteration %d : '
                         
                         '  dice_value_labeled: : %f'
                         
                         '  net_1_loss : %f'
                         '  net_2_loss : %f '
                         '  loss_pseudo_labeled : %f '
                         '  patch_loss: %f'
                         
                         '  net_1_c_loss: %f'
                         '  net_1_supervised_loss: %f'
                         '  net_1_consistency_loss : %f '
                         '  net_1_unsupervised_loss : %f '
                         
                         '  net_2_c_loss: %f'
                         '  net_2_supervised_loss: %f'
                         '  net_2_consistency_loss : %f '
                         '  net_2_unsupervised_loss : %f '
                         '  lr_ : %f'
                         % (iter_num,

                            dice_value_labeled,

                            loss_1.item(),
                            loss_2.item(),
                            loss_pseudo_labeled,
                            embed_p_factor * (patch_loss_mix + patch_loss) * 0.5,

                            embed_c_factor * (loss_embed_c_res + loss_embed_c_res_mixed) * 0.5,
                            supervised_loss_1,
                            (loss_diff_1 + consistency_loss_mix_1) * 0.5,
                            consistency_weight * (unsupervised_loss_1 + mix_loss_1) * 0.5,

                            embed_c_factor * (loss_embed_c_trans + loss_embed_c_trans_mixed) * 0.5,
                            supervised_loss_2,
                            (loss_diff_2 + consistency_loss_mix_2) * 0.5,
                            consistency_weight * (unsupervised_loss_2 + mix_loss_2) * 0.5,
                            lr_)
                         )


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        lr_ = self.args.base_lr * (1.0 - iter_num / self.args.max_iterations)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_

    def val(self, val_loader, snapshot_path, iter_num, logger):
        self.UC_seg.eval()
        avg_dice = 0.0


        for i_batch, sampled_batch in enumerate(val_loader):
            val_image, val_label = sampled_batch["image"].cuda(), sampled_batch["label"].cuda()
            metric_list = test_single_volume(self.args, val_image, val_label, self.UC_seg)
            metric_list = np.array(metric_list).astype("float32")
            metric_list = np.mean(metric_list, axis=0)
            avg_dice += metric_list[0]

        avg_dice = avg_dice / len(val_loader)


        logger.info('iteration %d : '
                     '  mean_dice : %f ' % (
                         iter_num, avg_dice))

        if avg_dice > self.best_performance_dice:
            self.best_performance_dice = avg_dice
            save_best = os.path.join(snapshot_path, 'best_model_' + str(iter_num) + '.pth')
            torch.save(self.UC_seg.state_dict(), save_best)


        self.UC_seg.train()




