import torch
import torch.nn as nn
from module.vnet_3D import VNet as vnet_3D
from module.unet_3D import unet_3D
from module.generate_pseudo_map_attention import Generate_pseudo_Transformer
from utils.UMIS import infer

class UC_Seg(nn.Module):
    def __init__(self, args):
        super(UC_Seg, self).__init__()
        feature_dim = 256
        self.args = args
        self.net_1 = unet_3D(in_channels=1, n_classes=args.num_classes)
        self.net_2 = vnet_3D(n_channels=1, n_classes=args.num_classes)
        self.generate_pseudo_simple = Generate_pseudo_Transformer(args)


    def forward(self, volume_batch):
        outputs_1, embedding_1 = self.net_1(volume_batch)
        outputs_2, embedding_2 = self.net_2(volume_batch)
        outputs_logits_1, outputs_logits_2 = outputs_1, outputs_2

        conf_pseudo_1 = self.args.num_classes / torch.sum(infer(outputs_logits_1) + 1, dim=1, keepdim=True)
        conf_pseudo_2 = self.args.num_classes / torch.sum(infer(outputs_logits_2) + 1, dim=1, keepdim=True)

        pseudo_map = self.generate_pseudo_simple(conf_pseudo_1, conf_pseudo_2, outputs_logits_1, outputs_logits_2)
        return outputs_1, embedding_1, outputs_2, embedding_2, pseudo_map
