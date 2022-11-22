# encoding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())
    dist_mat = dist_mat.mul(1 / qg_normdot)
    dist_mat = torch.clamp(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = torch.arccos(dist_mat)
    return dist_mat

def hard_example_mining_solflabel(dist_mat, labels, merge_method = 'cosine', thred = 0.5, return_inds=False):
    torch.autograd.set_detect_anomaly(True)
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    assert len(labels.shape) == 2
    N = dist_mat.size(0)

    if merge_method == 'cosine':
        dist = cosine_similarity(labels, labels)
        pos_weight = 1 - dist/ torch.pi
    
    is_pos = (pos_weight >= thred)
    is_neg = (pos_weight < thred)

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_pos = dist_mat*pos_weight
    dist_pos[is_neg] -= dist_pos[is_neg]    # avoid neg samples chose
    dist_ap, relative_p_inds = torch.max(
        dist_pos, 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_neg = dist_mat*(1-pos_weight)
    dist_neg[is_neg] += dist_neg[is_neg]    # avoid positive samples chose
    dist_an, relative_n_inds = torch.min(
        dist_neg, 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    # if return_inds:
    #     # shape [N, N]
    #     ind = (labels.new().resize_as_(labels)
    #            .copy_(torch.arange(0, N).long())
    #            .unsqueeze(0).expand(N, N))
    #     # shape [N, 1]
    #     p_inds = torch.gather(
    #         ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
    #     n_inds = torch.gather(
    #         ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
    #     # shape [N]
    #     p_inds = p_inds.squeeze(1)
    #     n_inds = n_inds.squeeze(1)
    #     return dist_ap, dist_an, p_inds, n_inds
    return dist_ap, dist_an



def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None, hard_factor=0.0, **kwargs):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, feat, target, normalize_feature=False, **kwargs):
        if normalize_feature:
            feat = normalize(feat, axis=-1)
        dist_mat = euclidean_dist(feat, feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, target)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss#, dist_ap, dist_an

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, **kwargs):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(input)
        target = torch.zeros(log_probs.size()).scatter_(1, target.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: target = target.cuda()
        target = (1 - self.epsilon) * target + self.epsilon / self.num_classes
        loss = (- target * log_probs).mean(0).sum()
        return loss


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


class WeightedRegularizedTriplet(object):

    def __init__(self, **kwargs):
        self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, feat, target, normalize_feature=True, **kwargs):
        if normalize_feature:
            feat = normalize(feat, axis=-1)
        dist_mat = euclidean_dist(feat, feat)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = target.expand(N, N).eq(target.expand(N, N).t()).float()
        is_neg = target.expand(N, N).ne(target.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        return loss#, furthest_positive, closest_negative


"""
Soft Triplet Loss
Following: https://github.com/idstcv/SoftTriple/blob/master/loss/SoftTriple.py
"""
class SoftTriple(nn.Module):
    def __init__(self,  dim, cN, K=10, la=20, gamma=0.1, tau=0.2, margin=0.01):
        super(SoftTriple, self).__init__()
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.fc = Parameter(torch.Tensor(dim, cN*K))
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).cuda()
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, feat, target):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = feat.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify

class SoftLabelTriplet(nn.Module):

    def __init__(self, margin=None, hard_factor=0.0, thred=0.5, **kwargs):
        super(SoftLabelTriplet, self).__init__()
        self.margin = margin
        self.hard_factor = hard_factor
        self.thred = thred
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, feat, target, normalize_feature=False, **kwargs):
        if normalize_feature:
            feat = normalize(feat, axis=-1)
        dist_mat = euclidean_dist(feat, feat)
        dist_ap, dist_an = hard_example_mining_solflabel(dist_mat, target, thred = self.thred)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss#, dist_ap, dist_an

"""
Soft Triplet Loss
Following: https://github.com/idstcv/SoftTriple/blob/master/loss/SoftTriple.py
Delete margin to use softlabel
"""
class SoftTripleSoftLabel(nn.Module):
    def __init__(self,  dim, cN, K=10, la=20, gamma=0.1, tau=0.2, margin=0.01):
        super(SoftTriple, self).__init__()
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.fc = Parameter(torch.Tensor(dim, cN*K))
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).cuda()
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, feat, target):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = feat.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        marginM = torch.zeros(simClass.shape).cuda()
        #marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify