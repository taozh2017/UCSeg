import torch

import torch.nn as nn

from module.vnet_2D import VNet
from module.unet_2D import UNet
from module.generate_pseudo_map_attention_2D import Generate_pseudo_Transformer
from utils.UMIS import infer


class UC_Seg(nn.Module):
    def __init__(self, args):
        super(UC_Seg, self).__init__()
        self.args = args
        self.net_1 = UNet(in_chns=args.in_channels, class_num=args.num_classes)
        self.net_2 = VNet(n_channels=args.in_channels, n_classes=args.num_classes)
        self.generate_pseudo_simple = Generate_pseudo_Transformer(args)


    def forward(self, volume_batch):
        outputs_1, embedding_1 = self.net_1(volume_batch)
        outputs_2, embedding_2 = self.net_2(volume_batch)
        outputs_logits_1, outputs_logits_2 = outputs_1, outputs_2
        # conf_pseudo_1 = 1 - self.args.num_classes / torch.sum(infer(outputs_logits_1) + 1, dim=1, keepdim=True)
        # conf_pseudo_2 = 1 - self.args.num_classes / torch.sum(infer(outputs_logits_2) + 1, dim=1, keepdim=True)
        conf_pseudo_1 = self.args.num_classes / torch.sum(infer(outputs_logits_1) + 1, dim=1, keepdim=True)
        conf_pseudo_2 = self.args.num_classes / torch.sum(infer(outputs_logits_2) + 1, dim=1, keepdim=True)
        pseudo_map = self.generate_pseudo_simple(conf_pseudo_1, conf_pseudo_2, outputs_logits_1, outputs_logits_2)

        return outputs_1, embedding_1, outputs_2, embedding_2, pseudo_map
























