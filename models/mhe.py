import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class MHE(nn.Module):
    """ reference: <Learning towards Minimum Hyperspherical Energy>"
    """
    def __init__(self, feat_dim=2048, n_views=3):
        super(MHE, self).__init__()
        self.feat_dim = feat_dim
        self.w = nn.Parameter(torch.Tensor(feat_dim, n_views))
        nn.init.xavier_normal_(self.w)

    def forward(self, x):
        # weight normalization
        with torch.no_grad():
            self.w.data = x
            self.w.data = F.normalize(self.w.data, dim=0)

        # mini-batch MHE loss for classifiers
        # sel_w = self.w[:,torch.unique(y)]
        sel_w = self.w
        gram_mat = torch.acos(torch.matmul(torch.transpose(sel_w, 0, 1), sel_w).clamp(-1.+1e-5, 1.-1e-5))
        shape_gram = gram_mat.size()
        MHE_loss = torch.sum(torch.triu(torch.pow(gram_mat, -2), diagonal=1))
        MHE_loss = MHE_loss / (shape_gram[0] * (shape_gram[0] - 1) * 0.5)

        return MHE_loss