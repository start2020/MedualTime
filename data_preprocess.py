import  torch.nn as nn
import torch
from torch.nn import functional as F




class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


########################################  Patch masking
def random_masking(xb, mask_ratio):
    # xb: [bs x num_patch x n_vars x patch_len]
    bs, L, nvars, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(bs, L, nvars, device=xb.device)  # noise in [0, 1], bs x L x nvars

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L x nvars]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep, :]  # ids_keep: [bs x len_keep x nvars]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1,
                                                                        D))  # x_kept: [bs x len_keep x nvars  x patch_len]

    # removed x
    x_removed = torch.zeros(bs, L - len_keep, nvars, D,
                            device=xb.device)  # x_removed: [bs x (L-len_keep) x nvars x patch_len]
    x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x nvars x patch_len]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1,
                                                                              D))  # x_masked: [bs x num_patch x nvars x patch_len]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L, nvars], device=x.device)  # mask: [bs x num_patch x nvars]
    mask[:, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch x nvars]
    return x_masked, x_kept, mask, ids_restore

def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
    tgt_len = patch_len + stride * (num_patch - 1)
    s_begin = seq_len - tgt_len

    xb = xb[:, s_begin:, :]  # xb: [bs x tgt_len x nvars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)  # xb: [bs x num_patch x n_vars x patch_len]
    return xb, num_patch

def patch_masking_dual(xb, args):
    # 用于在另外一个塔进行分割，然后放进gpt2
    """
    xb: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
    """
    xb_patch, num_patch = create_patch(xb, args.ts_emb_patch_len, args.ts_emb_patch_len)    # xb_patch: [bs x num_patch x n_vars x patch_len]
    xb_mask, _, mask, _ = random_masking(xb_patch, args.mask_ratio)   # xb_mask: [bs x num_patch x n_vars x patch_len]
    mask = mask.bool()    # mask: [bs x num_patch x n_vars]
    xb = xb_mask       # learner.xb: masked 4D tensor
    yb = xb_patch      # learner.yb: non-masked 4d tensor
    return xb, yb

def patch_masking(xb, args):
    """
    xb: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
    """
    xb_patch, num_patch = create_patch(xb, args.patch_len, args.stride)    # xb_patch: [bs x num_patch x n_vars x patch_len]
    xb_mask, _, mask, _ = random_masking(xb_patch, args.mask_ratio)   # xb_mask: [bs x num_patch x n_vars x patch_len]
    mask = mask.bool()    # mask: [bs x num_patch x n_vars]
    xb = xb_mask       # learner.xb: masked 4D tensor
    yb = xb_patch      # learner.yb: non-masked 4d tensor
    return xb, yb