import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from utils.losses import loss_diff1, loss_diff2

CE = torch.nn.BCELoss()
mse = torch.nn.MSELoss()


class Contrast_global_consistency(nn.Module):
    def __init__(self, args, feature_dim, K=1440, K_global=86400, T=0.3):
        super(Contrast_global_consistency, self).__init__()
        self.batch_size = args.batch_size
        self.labeled_bs = args.labeled_bs
        self.K = K
        self.K_global = K_global
        self.T = T
        self.args = args
        self.feature_dim = feature_dim
        self.criterion_mse = nn.MSELoss()
        self.criterion_ce = nn.CrossEntropyLoss()
        # self.projector_1 = nn.Sequential(nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1,
        #                                  stride=1, padding=0, bias=False),
        #                                  nn.BatchNorm2d(feature_dim),
        #                                  nn.LeakyReLU(inplace=True),
        #                                  nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1,
        #                                            stride=1, padding=0, bias=False),
        #                                  nn.BatchNorm2d(feature_dim),
        #                                  nn.LeakyReLU(inplace=True),
        #                                 )
        # self.projector_2 = nn.Sequential(nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1,
        #                                            stride=1, padding=0, bias=False),
        #                                  nn.BatchNorm2d(feature_dim),
        #                                  nn.LeakyReLU(inplace=True),
        #                                  nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1,
        #                                            stride=1, padding=0, bias=False),
        #                                  nn.BatchNorm2d(feature_dim),
        #                                  nn.LeakyReLU(inplace=True),
        #                                 )
        # self.linear_1 = nn.Sequential(
        #     nn.Linear(in_features=self.H*self.W*self.batch_size, out_features=int(self.H/2)*int(self.W/2)*self.batch_size),
        #     nn.GELU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=int(self.H/2)*int(self.W/2)*self.batch_size, out_features=3*3*self.batch_size),
        #     nn.Dropout(p=0.2),
        # )
        # self.linear_2 = nn.Sequential(
        #     nn.Linear(in_features=self.H*self.W*self.batch_size, out_features=int(self.H/2)*int(self.W/2)*self.batch_size),
        #     nn.GELU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=int(self.H/2)*int(self.W/2)*self.batch_size, out_features=3*3*self.batch_size),
        #     nn.Dropout(p=0.2),
        # )
        # self.linear_1_mix = nn.Sequential(
        #     nn.Linear(in_features=self.H*self.W*self.labeled_bs, out_features=int(self.H/2)*int(self.W/2)*self.labeled_bs),
        #     nn.GELU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=int(self.H/2)*int(self.W/2)*self.labeled_bs, out_features=3*3*self.labeled_bs),
        #     nn.Dropout(p=0.2),
        # )
        # self.linear_2_mix = nn.Sequential(
        #     nn.Linear(in_features=self.H*self.W*self.labeled_bs, out_features=int(self.H/2)*int(self.W/2)*self.labeled_bs),
        #     nn.GELU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=int(self.H/2)*int(self.W/2)*self.labeled_bs, out_features=3*3*self.labeled_bs),
        #     nn.Dropout(p=0.2),
        # )
        # create the queue
        self.register_buffer("classes_queue_feature_1", torch.randn(args.num_classes, feature_dim, K))
        self.classes_queue_feature_1 = F.normalize(self.classes_queue_feature_1, dim=0)
        self.register_buffer("classes_queue_feature_2", torch.randn(args.num_classes, feature_dim, K))
        self.classes_queue_feature_2 = F.normalize(self.classes_queue_feature_2, dim=0)
        # self.register_buffer("global_queue_feature", torch.randn(feature_dim, K_global))
        # self.global_queue_feature = F.normalize(self.global_queue_feature, dim=0)

        # queue init.
        self.register_buffer("classes_queue_ptr_1", torch.zeros(1, dtype=torch.long))
        self.register_buffer("classes_queue_ptr_2", torch.zeros(1, dtype=torch.long))
        # self.register_buffer("global_queue_ptr", torch.zeros(1, dtype=torch.long))

        self.pool7x7 = nn.AdaptiveAvgPool2d((7, 7))
        self.pool3x3 = nn.AdaptiveAvgPool2d((3, 3))
        self.pool1x1 = nn.AdaptiveAvgPool2d((1, 1))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, future, c=0, type=""):
        """update the queue."""
        if type == "feature_1":
            batch_size = future.shape[0]
            ptr = int(self.classes_queue_ptr_1)
            assert self.K % batch_size == 0
            if self.classes_queue_feature_1[c, :, ptr: ptr + batch_size].shape != future.T.shape:
                print("error")
                return
            self.classes_queue_feature_1[c, :, ptr: ptr + batch_size] = future.T
            ptr = (ptr + batch_size) % self.K  # move pointer
            self.classes_queue_ptr_1[0] = ptr

        elif type == "feature_2":
            # feature = torch.cat((future_1, future_2), dim=0)
            batch_size = future.shape[0]
            ptr = int(self.classes_queue_ptr_2)
            assert self.K % batch_size == 0
            if self.classes_queue_feature_2[c, :, ptr: ptr + batch_size].shape != future.T.shape:
                print("error")
                return
            self.classes_queue_feature_2[c, :, ptr: ptr + batch_size] = future.T
            ptr = (ptr + batch_size) % self.K  # move pointer
            self.classes_queue_ptr_2[0] = ptr

    def interpolation_save(self, feature, label, type=""):
        _, _, h, w = feature.shape
        label = (F.interpolate(label, size=(h, w), mode='bilinear', align_corners=False) > 0.5).float()
        for c in range(self.args.num_classes):
            class_feature = self.pool1x1(feature * label[:, c, ...].unsqueeze(1)).permute(0, 2, 3, 1).reshape(-1, self.feature_dim)
            self._dequeue_and_enqueue(class_feature, c, type=type)

    # def contrastiveLoss(self, pos, neg, temperature=0.1):
    #     """
    #     :param pos(Tensor): NxM positive similarity.
    #     :param neg(Tensor): Nxk negative similarity.
    #     :return dict[str, Tensor]:  A dictionary of loss components.
    #     """
    #     N = pos.size(0)
    #     losses_list = torch.zeros(pos.shape[1])
    #     for i in range(pos.shape[1]):
    #         logits = torch.cat((pos[:, i].unsqueeze(-1), neg), dim=1)
    #         logits /= temperature
    #         labels = torch.zeros((N,), dtype=torch.long).cuda()
    #         losses = self.criterion_ce(logits, labels)
    #         losses_list[i] = losses
    #     return losses_list.mean()

    def contrastiveLoss(self, pos, neg, temperature=0.1):
        """
        :param pos(Tensor): NxM positive similarity.
        :param neg(Tensor): Nxk negative similarity.
        :return dict[str, Tensor]:  A dictionary of loss components.
        """
        logits = torch.cat((pos, neg), dim=1)
        logits /= temperature
        labels = torch.zeros((pos.size(0),), dtype=torch.long).cuda()
        # labels = torch.arange(pos.size(0)).to(pos.device)
        losses = self.criterion_ce(logits, labels)
        return losses
    
    def patch_contrastiveLoss(self, pos, neg, temperature=0.1):
        """
        :param pos(Tensor): NxM positive similarity.
        :param neg(Tensor): Nxk negative similarity.
        :return dict[str, Tensor]:  A dictionary of loss components.
        """
        logits = torch.cat((pos, neg), dim=1)
        logits /= temperature
        labels = torch.zeros((pos.size(0),), dtype=torch.long).cuda()
        # labels = torch.arange(pos.size(0)).to(pos.device)
        losses = self.criterion_ce(logits, labels)
        return losses

    def contrast(self, feature_1, feature_2, predict_1, predict_2, is_mixed_image):
        # feature_pro_1 = self.projector_1(feature_1).permute(0, 2, 3, 1).reshape(-1, self.feature_dim)
        # feature_pro_2 = self.projector_2(feature_2).permute(0, 2, 3, 1).reshape(-1, self.feature_dim)
        # if not is_mixed_image:
        #     feature_pro_1 = self.linear_1(feature_pro_1.permute(1, 0)).permute(1, 0)
        #     feature_pro_2 = self.linear_2(feature_pro_2.permute(1, 0)).permute(1, 0)
        # else:
        #     feature_pro_1 = self.linear_1_mix(feature_pro_1.permute(1, 0)).permute(1,0)
        #     feature_pro_2 = self.linear_2_mix(feature_pro_2.permute(1, 0)).permute(1,0)
        #
        # feature_pro_1 = nn.functional.normalize(feature_pro_1, dim=-1)
        # feature_pro_2 = nn.functional.normalize(feature_pro_2, dim=-1)

        # q_dense = torch.einsum('nc,ck->nk', [feature_pro_1, self.global_queue_feature.clone().detach()])
        # k_dense = torch.einsum('nc,ck->nk', [feature_pro_2, self.global_queue_feature.clone().detach()])
        #
        # loss_s_1 = self.criterion_mse(q_dense, k_dense.clone().detach())
        # loss_s_2 = self.criterion_mse(k_dense, q_dense.clone().detach())

        # loss_s_1 = loss_diff1(q_dense, k_dense.clone().detach())
        # loss_s_2 = loss_diff2(k_dense, q_dense.clone().detach())

        b, c, h, w = feature_1.shape

        patches = 3 * 3
        patch_loss = torch.zeros(patches)
        feature_pool_1 = self.pool3x3(feature_1).reshape(b, c, -1)
        feature_pool_2 = self.pool3x3(feature_2).reshape(b, c, -1)

        for i in range(patches):
            # a = feature_pool_1[:, :, i]
            feature_pos = torch.einsum('nc,nc->n', [feature_pool_1[:, :, i], feature_pool_2[:, :, i]]).unsqueeze(-1)
            neg_sample = torch.cat([feature_pool_2[:, :, :i], feature_pool_2[:, :, i+1:]], dim=-1).permute(1, 0, 2).reshape(self.feature_dim, -1)
            feature_neg = torch.einsum('nc, ck->nk', [feature_pool_1[:, :, i], neg_sample])
            patch_loss[i] = self.patch_contrastiveLoss(feature_pos, feature_neg, self.T)

        predict_1 = (F.interpolate(predict_1, size=(h, w), mode='bilinear', align_corners=False) > 0.5).float()
        predict_2 = (F.interpolate(predict_2, size=(h, w), mode='bilinear', align_corners=False) > 0.5).float()

        classes_loss_1 = torch.zeros(self.args.num_classes)
        classes_loss_2 = torch.zeros(self.args.num_classes)
        # classes_loss_1 = 0
        # classes_loss_2 = 0
        for c in range(self.args.num_classes):
            cur_feature_1_prototype = self.pool1x1(feature_1 * predict_1[:, c, ...].unsqueeze(1)).permute(0, 2, 3, 1).reshape(-1, self.feature_dim)
            cur_feature_2_prototype = self.pool1x1(feature_2 * predict_2[:, c, ...].unsqueeze(1)).permute(0, 2, 3, 1).reshape(-1, self.feature_dim)
            pos_queue_feature_1 = self.classes_queue_feature_1[c, ...].clone().detach()
            pos_queue_feature_2 = self.classes_queue_feature_2[c, ...].clone().detach()
            if c != self.args.num_classes - 1:
                neg_queue_feature_1 = torch.cat(
                    (self.classes_queue_feature_1[:c, ...].clone().detach(), self.classes_queue_feature_1[c + 1:, ...].clone().detach()),
                    dim=0).permute(1, 0, 2).contiguous().view(self.feature_dim, -1)
                neg_queue_feature_2 = torch.cat(
                    (self.classes_queue_feature_2[:c, ...].clone().detach(), self.classes_queue_feature_2[c + 1:, ...].clone().detach()),
                    dim=0).permute(1, 0, 2).contiguous().view(self.feature_dim, -1)
            else:
                neg_queue_feature_1 = self.classes_queue_feature_1[:c, ...].permute(1, 0, 2).contiguous().view(self.feature_dim, -1).clone().detach()
                neg_queue_feature_2 = self.classes_queue_feature_2[:c, ...].permute(1, 0, 2).contiguous().view(self.feature_dim, -1).clone().detach()
            # cur_feature_prototype = torch.cat((cur_feature_1_prototype, cur_feature_2_prototype), dim=0)
            # pos_queue_feature_1 = torch.mean(pos_queue_feature_1, dim=1).unsqueeze(1)
            # pos_queue_feature_2 = torch.mean(pos_queue_feature_2, dim=1).unsqueeze(1)
            if is_mixed_image:
                # pos_queue_feature_1 = torch.mean(pos_queue_feature_1.view(self.feature_dim, self.args.labeled_bs, -1), dim=-1).permute(1, 0)
                # pos_queue_feature_2 = torch.mean(pos_queue_feature_2.view(self.feature_dim, self.args.labeled_bs, -1),dim=-1).permute(1, 0)
                pos_queue_feature_1 = torch.mean(pos_queue_feature_1, dim=1).unsqueeze(1).repeat(1, self.args.labeled_bs).permute(1, 0)
                pos_queue_feature_2 = torch.mean(pos_queue_feature_2, dim=1).unsqueeze(1).repeat(1, self.args.labeled_bs).permute(1, 0)
            else:
                # pos_queue_feature_1 = torch.mean(pos_queue_feature_1.view(self.feature_dim, self.args.batch_size, -1), dim=-1).permute(1, 0)
                # pos_queue_feature_2 = torch.mean(pos_queue_feature_2.view(self.feature_dim, self.args.batch_size, -1), dim=-1).permute(1, 0)
                pos_queue_feature_1 = torch.mean(pos_queue_feature_1, dim=1).unsqueeze(1).repeat(1, self.args.batch_size).permute(1, 0)
                pos_queue_feature_2 = torch.mean(pos_queue_feature_2, dim=1).unsqueeze(1).repeat(1, self.args.batch_size).permute(1, 0)

            # feature_sim_pos_1 = torch.einsum('bf,fk->bk', [cur_feature_1_prototype, pos_queue_feature_1])
            # feature_sim_pos_2 = torch.einsum('bf,fk->bk', [cur_feature_2_prototype, pos_queue_feature_2])
            feature_sim_pos_1 = torch.einsum('nc,nc->n', [cur_feature_1_prototype, pos_queue_feature_1]).unsqueeze(-1)
            feature_sim_pos_2 = torch.einsum('nc,nc->n', [cur_feature_2_prototype, pos_queue_feature_2]).unsqueeze(-1)

            feature_sim_neg_1 = torch.einsum('nc,ck->nk', [cur_feature_1_prototype, neg_queue_feature_1])
            feature_sim_neg_2 = torch.einsum('nc,ck->nk', [cur_feature_2_prototype, neg_queue_feature_2])
            loss_c_1 = self.contrastiveLoss(feature_sim_pos_1, feature_sim_neg_1, self.T)
            loss_c_2 = self.contrastiveLoss(feature_sim_pos_2, feature_sim_neg_2, self.T)
            classes_loss_1[c] = loss_c_1
            classes_loss_2[c] = loss_c_2
            # classes_loss_1 += self.contrastiveLoss(feature_sim_pos_1, feature_sim_neg_1, self.T)
            # classes_loss_2 += self.contrastiveLoss(feature_sim_pos_2, feature_sim_neg_2, self.T)
        return patch_loss.mean(), classes_loss_1.mean(), classes_loss_2.mean()
        # return loss_s_1, loss_s_2, classes_loss_1, classes_loss_2

    def forward(self, stat, feature_1=None, feature_2=None, predict_1=None, predict_2=None, label=None, is_mixed_image=False, save_type=""):
        if stat == "contrast":
            patch_loss, loss_embed_c_1, loss_embed_c_2 = self.contrast(feature_1, feature_2, predict_1, predict_2, is_mixed_image)
            return patch_loss, loss_embed_c_1, loss_embed_c_2
        else:
            if save_type == "feature_1":
                self.interpolation_save(feature_1, label, type=save_type)
            elif save_type == "feature_2":
                self.interpolation_save(feature_2, label, type=save_type)


if __name__ == '__main__':
    for i in range(100):
        q = torch.randn((12, 256, 14, 14)).cuda()
        k = torch.randn((12, 256, 14, 14)).cuda()
        predict = torch.randint(0, 2, size=(12, 2, 224, 224)).float().cuda()
        label = torch.randint(0, 2, size=(12, 2, 224, 224)).float().cuda()
        model = Contrast_global_consistency()
        loss_1 = model("contrast", q, predict, label)
        print(loss_1)







































