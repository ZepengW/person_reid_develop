# encoding: utf-8

import numpy as np
import torch

from .rank import evaluate_rank
import torch.nn.functional as F
from utils.dist_compute import build_dist
from utils.re_ranking import re_ranking, re_ranking_optimize
import csv
import os
from datetime import datetime

class Evaluator(object):
    def __init__(self, **kwargs):
        self.metric = kwargs.get('metric', 'euclidean')
        self.k1 = kwargs.get('k1', 20)
        self.k2 = kwargs.get('k2', 6)
        self.lambda_value = kwargs.get('lambda_value', 0.3)
        self.use_metric_cuhk03 = kwargs.get('use_metric_cuhk03', False)
        self.norm = kwargs.get('norm', False)
        self.save_dir = kwargs.get('save_dir', 'result_infer')
        self.mode = kwargs.get('mode', None)

    def __call__(self, feat_q, feat_g, q_pids, g_pids, q_camids, g_camids, max_rank=50, re_ranking=False):
        print("retrival evaluate")
        if self.norm:
            feat = F.normalize(torch.cat([feat_q, feat_g], dim=0), dim=1)
            feat_q = feat[:feat_q.shape[0]]
            feat_g = feat[feat_q.shape[0]:]
        print("compute dist mat")
        dist = build_dist(feat_q, feat_g, metric=self.metric)
        if re_ranking:
            print("Re-Ranking")
            dist_qq = build_dist(feat_q, feat_q, metric=self.metric)
            dist_gg = build_dist(feat_g, feat_g, metric=self.metric)
            dist = re_ranking_optimize(dist, dist_qq, dist_gg,\
                k1=self.k1, k2=self.k2, lambda_value=self.lambda_value)
        print("calculate metric")
        cmc, all_AP, all_INP = evaluate_rank(dist, q_pids, g_pids, q_camids, g_camids,
                                             max_rank=max_rank, use_metric_cuhk03=self.use_metric_cuhk03)
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        return cmc, mAP, mINP
    
    def infer(self, feat_q, feat_g, name_q=None, name_g=None, max_rank=50, re_ranking=False):
        print("retrival evaluate")
        if self.norm:
            feat = F.normalize(torch.cat([feat_q, feat_g], dim=0), dim=1)
            feat_q = feat[:feat_q.shape[0]]
            feat_g = feat[feat_q.shape[0]:]
        print("compute dist mat")
        dist = build_dist(feat_q, feat_g, metric=self.metric)
        if re_ranking:
            print("Re-Ranking")
            dist_qq = build_dist(feat_q, feat_q, metric=self.metric)
            dist_gg = build_dist(feat_g, feat_g, metric=self.metric)
            dist = re_ranking_optimize(dist, dist_qq, dist_gg,\
                k1=self.k1, k2=self.k2, lambda_value=self.lambda_value)
        if self.mode == 'FWreID':
            indices = np.argsort(dist, axis=1)
            now = datetime.now().strftime("%Y%m%d:%H%M%S")
            save_path = os.path.join(self.save_dir, f'indices-{now}.csv')
            with open(save_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(indices)