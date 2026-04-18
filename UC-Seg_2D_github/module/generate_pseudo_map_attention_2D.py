import os
import cv2
import torch
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def feature_vis(feats):  # feaats形状: [b,c,h,w]
    output_shape = (224, 224)  # 输出形状
    channel_mean = torch.mean(feats, dim=1, keepdim=True)  # channel_max,_ = torch.max(feats,dim=1,keepdim=True)
    channel_mean = F.interpolate(channel_mean, size=output_shape, mode='bilinear', align_corners=False)
    channel_mean = channel_mean.squeeze(0).squeeze(0).detach().cpu().numpy()  # 四维压缩为二维
    channel_mean = (((channel_mean - np.min(channel_mean)) / (np.max(channel_mean) - np.min(channel_mean))) * 255).astype(np.uint8)
    savedir = 'C:/Users/admin/Desktop/UC-Seg/Result_images/feats/'
    if not os.path.exists(savedir + 'feature_vis'): os.makedirs(savedir + 'feature_vis')
    channel_mean = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)
    cv2.imwrite(savedir + 'feature_vis/' + '0.png', channel_mean)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)



class Generate_pseudo_simple(torch.nn.Module):
    def __init__(self, num_classes, in_channels):
        super(Generate_pseudo_simple, self).__init__()
        # in_channels = num_classes
        in_channels = in_channels * 2
        # in_channels = num_classes * 3
        conv1_channels = 16
        conv2_channels = 32
        conv3_channels = 64
        self.out_conv_channels = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=conv1_channels, kernel_size=1,
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(conv1_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=1,
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(conv2_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv2_channels, out_channels=conv3_channels, kernel_size=1,
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(conv3_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                # in_channels=conv3_channels, out_channels=self.out_conv_channels, kernel_size=1,
                in_channels=conv3_channels, out_channels=conv2_channels, kernel_size=1,
                stride=1, padding=0, bias=False
            ),
            # nn.BatchNorm2d(self.out_conv_channels),
            nn.BatchNorm2d(conv2_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # add new
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv2_channels, out_channels=conv1_channels, kernel_size=1,
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(conv1_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv1_channels, out_channels=self.out_conv_channels, kernel_size=1,
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.out_conv_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # feature_vis(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


def show_soft(attn_p1p2):
    import matplotlib.pyplot as plt
    attn_p1p2 = attn_p1p2.squeeze(0).detach().cpu().numpy()
    fig, axes = plt.subplots(1, 8, figsize=(15, 10))
    for i in range(attn_p1p2.shape[0]):
        axes[i].imshow(attn_p1p2[i, :, :], cmap="gray")
        axes[i].set_axis_off()
    plt.show()


def show_attn(attn_p1p2):
    import matplotlib.pyplot as plt
    attn_p1p2 = attn_p1p2.squeeze(0).squeeze(0).detach().cpu().numpy()
    plt.imshow(attn_p1p2)
    # plt.imshow(attn_p1p2, cmap="gray")
    plt.show()



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads

        self.w_q = nn.Linear(dim, inner_dim)
        self.w_k = nn.Linear(dim, inner_dim)
        self.w_v = nn.Linear(dim, inner_dim)

        self.scale = dim_head ** -0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        project_out = not (num_heads == 1 and dim_head == dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, p1, p2):
        q_p1 = self.w_q(p1)
        k_p2 = self.w_k(p2)
        v_p2 = self.w_v(p2)
        q_p1 = rearrange(q_p1, 'b n (h d) -> b h n d', h=self.num_heads)
        k_p2 = rearrange(k_p2, 'b n (h d) -> b h n d', h=self.num_heads)
        v_p2 = rearrange(v_p2, 'b n (h d) -> b h n d', h=self.num_heads)

        attn_p1p2 = einsum('b h i d, b h j d -> b h i j', q_p1, k_p2) * self.scale
        attn_p1p2 = attn_p1p2.softmax(dim=-1)
        # show(attn_p1p2)
        attn_p1p2 = einsum('b h i j, b h j d -> b h i d', attn_p1p2, v_p2)
        attn_p1p2 = rearrange(attn_p1p2, 'b h n d -> b n (h d)')

        attn_p1p2 = self.to_out(attn_p1p2)
        return attn_p1p2


class Cross_Attention_block(nn.Module):
    def __init__(self, input_size, in_channels, patch_size=14, num_heads=8, channel_attn_drop=0.1, pos_embed=True, dim=96, dim_head=64, hid_dim=384):
        super(Cross_Attention_block, self).__init__()
        self.patch_size = patch_size

        assert input_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (input_size // patch_size) ** 2

        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim)
        )
        self.dropout = nn.Dropout(channel_attn_drop)

        self.attn = Attention(dim, num_heads, dim_head, channel_attn_drop)

        if pos_embed:
            # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, dim))
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))
        else:
            self.pos_embed = None

        self.MLP = FeedForward(dim, hid_dim)

        self.to_out = nn.Sequential(
            nn.Linear(dim, patch_dim),
            nn.Dropout(channel_attn_drop),
            Rearrange('b (h w) (p1 p2 c)-> b c (h p1) (w p2) ', h=(input_size // patch_size), w=(input_size // patch_size),  p1=patch_size, p2=patch_size),
        )

    def forward(self, p1, p2):

        p1 = self.to_patch_embedding(p1)
        p2 = self.to_patch_embedding(p2)
        _, n, _ = p1.shape       # n表示每个块的空间分辨率

        if self.pos_embed is not None:
            p1 = p1 + self.pos_embed
            p2 = p2 + self.pos_embed
        p1 = self.dropout(p1)
        p2 = self.dropout(p2)

        attn_p1p2 = self.attn(p1, p2)

        attn_p1p2 = self.MLP(attn_p1p2) + attn_p1p2
        attn_p1p2 = self.to_out(attn_p1p2)
        # show_attn(attn_p1p2)
        return attn_p1p2


class Generate_pseudo_Transformer(torch.nn.Module):
    def __init__(self, args):
        super(Generate_pseudo_Transformer, self).__init__()
        # in_channels = args.num_classes
        # self.cross_attn = Cross_Attention_block(args.img_size[0], in_channels)
        # self.final_head = Generate_pseudo_simple(in_channels)
        self.final_head = Generate_pseudo_simple(args.num_classes, args.num_classes)
        self.cross_attn = Cross_Attention_block(args.img_size[0], 1, args.patch_size)

    def forward(self, c1, c2, p1, p2):
        # feature_vis(p1)
        c1_c2 = self.cross_attn(c1, c2)
        conf_p1 = c1_c2 * p1
        conf_p2 = c1_c2 * p2
        p1_p2 = torch.cat((conf_p1, conf_p2), dim=1)
        output = self.final_head(p1_p2)
        return output

    # def forward(self, c1_c2_p1_p2):
    #     c1, c2, p1, p2 = c1_c2_p1_p2[:, 0, ...].unsqueeze(1), c1_c2_p1_p2[:, 1, ...].unsqueeze(1), c1_c2_p1_p2[:, 2:4, ...], c1_c2_p1_p2[:, 4:6, ...]
    #     c1_c2 = self.cross_attn(c1, c2)
    #     conf_p1 = c1_c2 * p1
    #     conf_p2 = c1_c2 * p2
    #     p1_p2 = torch.cat((conf_p1, conf_p2), dim=1)
    #     output = self.final_head(p1_p2)
    #     return output



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2,
                        help='output channel of network')
    parser.add_argument('--img_size', type=list, default=[224, 224],
                        help='patch size of network input')
    parser.add_argument('--labeled_bs', type=int, default=12,
                        help='labeled_batch_size per gpu')
    args = parser.parse_args()
    u1 = torch.randn((12, 1, 224, 224)).cuda()
    u2 = torch.randn((12, 1, 224, 224)).cuda()
    u1 = torch.cat((u1, u1), dim=1)
    u2 = torch.cat((u2, u2), dim=1)
    p1 = torch.randn((12, 2, 224, 224)).cuda()
    p2 = torch.randn((12, 2, 224, 224)).cuda()
    model = Generate_pseudo_Transformer(args).cuda()

    output = model(u1, u2, p1, p2)
    print(output.shape)















