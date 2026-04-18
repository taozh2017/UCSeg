import torch
import torch.nn as nn
import torch.nn.functional as F

CE = torch.nn.BCELoss()
mse = torch.nn.MSELoss()

class Contrast_global_consistency(nn.Module):
    def  __init__(self, args, feature_dim, K=144, K_global=4320, T=0.3):
        super(Contrast_global_consistency, self).__init__()
        self.H = int(args.patch_size[0]/16)
        self.W = int(args.patch_size[1]/16)
        self.D = int(args.patch_size[2]/16)
        self.batch_size = args.batch_size
        self.labeled_bs = args.labeled_bs
        self.K = K
        self.K_global = K_global
        self.T = T
        self.args = args
        self.feature_dim = feature_dim
        self.criterion_mse = nn.MSELoss()
        self.criterion_ce = nn.CrossEntropyLoss()


        # create the queue
        self.register_buffer("classes_queue_feature_1", torch.randn(args.num_classes, feature_dim, K))
        self.classes_queue_feature_1 = F.normalize(self.classes_queue_feature_1, dim=1)
        self.register_buffer("classes_queue_feature_2", torch.randn(args.num_classes, feature_dim, K))
        self.classes_queue_feature_2 = F.normalize(self.classes_queue_feature_2, dim=1)


        # queue init.
        self.register_buffer("classes_queue_ptr_1", torch.zeros(1, dtype=torch.long))
        self.register_buffer("classes_queue_ptr_2", torch.zeros(1, dtype=torch.long))

        self.pool3x3x3 = nn.AdaptiveAvgPool3d((3, 3, 3))
        self.pool1x1x1 = nn.AdaptiveAvgPool3d((1, 1, 1))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, feature, c=0, type=""):
        """update the queue."""
        if type == "feature_1":
            batch_size = feature.shape[0]
            ptr = int(self.classes_queue_ptr_1)
            assert self.K % batch_size == 0
            if self.classes_queue_feature_1[c, :, ptr: ptr + batch_size].shape != feature.T.shape:
                print("error")
                return
            self.classes_queue_feature_1[c, :, ptr: ptr + batch_size] = feature.T
            ptr = (ptr + batch_size) % self.K  # move pointer
            self.classes_queue_ptr_1[0] = ptr
        elif type == "feature_2":
            batch_size = feature.shape[0]
            ptr = int(self.classes_queue_ptr_2)
            assert self.K % batch_size == 0
            if self.classes_queue_feature_2[c, :, ptr: ptr + batch_size].shape != feature.T.shape:
                print("error")
                return
            self.classes_queue_feature_2[c, :, ptr: ptr + batch_size] = feature.T
            ptr = (ptr + batch_size) % self.K  # move pointer
            self.classes_queue_ptr_2[0] = ptr

    def interpolation_save(self, feature, label, type=""):
        _, _, h, w, d = feature.shape
        label = (F.interpolate(label, size=(h, w, d), mode='trilinear', align_corners=False) > 0.5).float()
        for c in range(self.args.num_classes):
            class_feature = self.pool1x1x1(feature * label[:, c, ...].unsqueeze(1)).permute(0, 2, 3, 4, 1).reshape(-1, self.feature_dim)
            self._dequeue_and_enqueue(class_feature, c, type=type)

    def contrastiveLoss(self, pos, neg, temperature=0.1):
        """
        :param pos(Tensor): NxM positive similarity.
        :param neg(Tensor): Nxk negative similarity.
        :return dict[str, Tensor]:  A dictionary of loss components.
        """
        logits = torch.cat((pos, neg), dim=1)
        logits /= temperature
        labels = torch.zeros((pos.size(0),), dtype=torch.long).cuda()
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
        losses = self.criterion_ce(logits, labels)
        return losses

    def contrast(self, feature_1, feature_2, predict_1, predict_2, is_mixed_image):

        b, c, h, w, d = feature_1.shape

        patches = 3 * 3 * 3
        patch_loss = torch.zeros(patches)
        feature_pool_1 = self.pool3x3x3(feature_1).reshape(b, c, -1)
        feature_pool_2 = self.pool3x3x3(feature_2).reshape(b, c, -1)

        for i in range(patches):
            # a = feature_pool_1[:, :, i]
            feature_pos = torch.einsum('nc,nc->n', [feature_pool_1[:, :, i], feature_pool_2[:, :, i]]).unsqueeze(-1)
            neg_sample = torch.cat([feature_pool_2[:, :, :i], feature_pool_2[:, :, i+1:]], dim=-1).permute(1, 0, 2).reshape(self.feature_dim, -1)
            feature_neg = torch.einsum('nc, ck->nk', [feature_pool_1[:, :, i], neg_sample])
            patch_loss[i] = self.patch_contrastiveLoss(feature_pos, feature_neg, self.T)

        predict_1 = (F.interpolate(predict_1, size=(h, w, d), mode='trilinear', align_corners=False) > 0.5).float()
        predict_2 = (F.interpolate(predict_2, size=(h, w, d), mode='trilinear', align_corners=False) > 0.5).float()

        classes_loss_1 = torch.zeros(self.args.num_classes)
        classes_loss_2 = torch.zeros(self.args.num_classes)

        for c in range(self.args.num_classes):
            cur_feature_1_prototype = self.pool1x1x1(feature_1 * predict_1[:, c, ...].unsqueeze(1)).permute(0, 2, 3, 4, 1).reshape(-1, self.feature_dim)
            cur_feature_2_prototype = self.pool1x1x1(feature_2 * predict_2[:, c, ...].unsqueeze(1)).permute(0, 2, 3, 4, 1).reshape(-1, self.feature_dim)
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

            if is_mixed_image:

                pos_queue_feature_1 = torch.mean(pos_queue_feature_1, dim=1).unsqueeze(1).repeat(1, self.args.labeled_bs).permute(1, 0)
                pos_queue_feature_2 = torch.mean(pos_queue_feature_2, dim=1).unsqueeze(1).repeat(1, self.args.labeled_bs).permute(1, 0)
            else:

                pos_queue_feature_1 = torch.mean(pos_queue_feature_1, dim=1).unsqueeze(1).repeat(1, self.args.batch_size).permute(1, 0)
                pos_queue_feature_2 = torch.mean(pos_queue_feature_2, dim=1).unsqueeze(1).repeat(1, self.args.batch_size).permute(1, 0)


            feature_sim_pos_1 = torch.einsum('nc,nc->n', [cur_feature_1_prototype, pos_queue_feature_1]).unsqueeze(-1)
            feature_sim_pos_2 = torch.einsum('nc,nc->n', [cur_feature_2_prototype, pos_queue_feature_2]).unsqueeze(-1)

            feature_sim_neg_1 = torch.einsum('nc,ck->nk', [cur_feature_1_prototype, neg_queue_feature_1])
            feature_sim_neg_2 = torch.einsum('nc,ck->nk', [cur_feature_2_prototype, neg_queue_feature_2])
            loss_c_1 = self.contrastiveLoss(feature_sim_pos_1, feature_sim_neg_1, self.T)
            loss_c_2 = self.contrastiveLoss(feature_sim_pos_2, feature_sim_neg_2, self.T)
            classes_loss_1[c] = loss_c_1
            classes_loss_2[c] = loss_c_2

        return patch_loss.mean(), classes_loss_1.mean(), classes_loss_2.mean()


    def forward(self, stat, feature_1=None, feature_2=None, predict_1=None, predict_2=None, label=None, is_mixed_image=False, save_type=""):
        if stat == "contrast":
            patch_loss, loss_embed_c_1, loss_embed_c_2 = self.contrast(feature_1, feature_2, predict_1, predict_2, is_mixed_image)
            return patch_loss, loss_embed_c_1, loss_embed_c_2
        else:
            if save_type == "feature_1":
                self.interpolation_save(feature_1, label, type=save_type)
            elif save_type == "feature_2":
                self.interpolation_save(feature_2, label, type=save_type)












































