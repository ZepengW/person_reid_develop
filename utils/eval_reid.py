# encoding: utf-8

import numpy as np
from .rank import evaluate_rank
import torch.nn.functional as F
from utils.dist_compute import build_dist
from utils.re_ranking import re_ranking

class Evaluator(object):
    def __init__(self, metric='euclidean', use_metric_cuhk03=False, k1=20, k2=6, lambda_value=0.2):
        self.metric = metric
        self.k1 = k1
        self.k2 = k2
        self.lambda_value = lambda_value
        self.use_metric_cuhk03 = use_metric_cuhk03

    def __call__(self, feat_q, feat_g, q_pids, g_pids, q_camids, g_camids, max_rank=50, re_ranking=False):
        print("retrival evaluate")
        print("compute dist mat")
        dist = build_dist(feat_q, feat_g, metric=self.metric)
        print("calculate metric")
        cmc, all_AP, all_INP = evaluate_rank(dist, q_pids, g_pids, q_camids, g_camids,
                                             max_rank=max_rank, use_metric_cuhk03=self.use_metric_cuhk03)
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        if re_ranking:
            print("Re-Ranking")
            # todo: re-ranking
        else:
            return [cmc, mAP, mINP]
