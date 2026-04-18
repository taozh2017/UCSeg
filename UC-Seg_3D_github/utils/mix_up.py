import torch
import numpy as np
import torch.nn.functional as F
from skimage.measure import label


def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    return torch.Tensor(batch_list).cuda()

def get_cut_mask(out, thres=0.5, nms=0):
    probs = torch.nn.Sigmoid()(out)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks


def generate_mask_3D(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0], img.shape[1], img.shape[2], img.shape[3], img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    # patch_x, patch_y, patch_z = int(img_x*2/3), int(img_y*2/3), int(img_z)
    patch_x, patch_y, patch_z = int(img_x*2/3), int(img_y*2/3), int(img_z*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    d = np.random.randint(0, img_z - patch_z)
    mask[w:w+patch_x, h:h+patch_y, d:d+patch_z] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y, d:d+patch_z] = 0
    # mask[w:w+patch_x, h:h+patch_y, ...] = 0
    # loss_mask[:, w:w+patch_x, h:h+patch_y, ...] = 0
    return mask.long(), loss_mask.long()
