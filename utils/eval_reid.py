# encoding: utf-8

import numpy as np
import cv2
import math
import random
import torch
import os

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, q_img_paths=None, g_img_paths=None):
    """
    Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    output the detail result: img path
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    # vis result
    res_vis = []  # data format: ((query_img_path, gallery_img_path_1, gallery_img_path_2, ...), res_label_list)
    if q_img_paths is not None and g_img_paths is not None:
        q_img_paths = np.array(q_img_paths)
        g_img_paths = np.array(g_img_paths)
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        if (q_camid == -1).all():
            remove = (g_pids[order] == q_pid)
            remove[:] = False
        else:
            # remove gallery samples that have the same pid and camid with query
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        # new add for not delete sample with same pid and cid
        # list = [True] * np.shape(order)[0]
        # keep = np.array(list)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matchesN
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        # record imgs' path of result
        if q_img_paths is not None and g_img_paths is not None:
            # first element is query image path, others are ranked gallery image path
            r = np.concatenate([q_img_paths[q_idx:q_idx+1], g_img_paths[order][keep]])
            pids = []
            pids.append(q_pid)
            pids += g_pids[order][keep].tolist()
            res_vis.append((r, orig_cmc, pids))

        cmc = orig_cmc.cumsum()
        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP, res_vis


def visualize_result(res_vis, max_rank=10, size=(50, 100), sample_method='random', sample_num = 20, writer=None):
    '''
    visualize rank result generate by eval_func
    :param res_vis:
    :param max_rank:
    :param size:
    :param sample_method:   random/list_false
    :param writer:
    :return:
    '''
    if writer is None:
        return
    imgs = []
    r1_flag_l = []
    for (img_paths, labels, pids) in res_vis:
        img_vis, r1_flag = conbine_imgs(img_paths,size,labels,pids, max_rank)
        imgs.append(img_vis)
        r1_flag_l.append(r1_flag)
    if sample_method == 'random':
        imgs_selected_vis = random.sample(imgs, sample_num)
    elif sample_method == 'list_false':
        imgs_selected_vis = [imgs[i] for i in range(len(r1_flag_l)) if r1_flag_l[i] == False]
    elif sample_method == 'all':
        imgs_selected_vis = imgs
    else:
        imgs_selected_vis = []
    split_num = math.ceil(len(imgs_selected_vis) / 10)
    for i in range(split_num):
        imgs_vis = torch.cat(imgs_selected_vis[i*10:i*10 + 10], dim=1)
        writer.add_image(sample_method, imgs_vis, i)



def conbine_imgs(img_paths, size, labels, pids, max_rank = 10):
    '''
    combine query img and ranked gallery imgs to a total img
    :param img_paths:
    :param size:
    :param labels:
    :return:
    '''
    # read query img
    q_img_path = img_paths[0]
    q_img = cv2.imread(q_img_path)
    q_img = cv2.resize(q_img, size)
    # add query text label
    text_bg = np.zeros([20, q_img.shape[1], q_img.shape[2]])
    q_pid = pids[0]
    text_bg = cv2.putText(text_bg, str(q_pid), (0, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    q_img = np.concatenate([q_img, text_bg], 0)
    img_res = []
    img_res.append(q_img)
    # append space
    space = np.zeros([q_img.shape[0], int(q_img.shape[1] / 2), q_img.shape[2]], dtype='uint8')
    space[:] = 255
    img_res.append(space)
    rank1_success_marks = False # this is used to mark the rank 1 result is True or False
    for i, g_img_path in enumerate(img_paths[1:max_rank+1]):
        g_img = cv2.imread(g_img_path)
        g_img = cv2.resize(g_img, size)
        if labels[i] == 1:
            # true
            g_img = cv2.rectangle(g_img,(0, 0),(size[0]-1,size[1]-1),(0, 255, 0),2)
            flag = True
        else:
            # wrong
            g_img = cv2.rectangle(g_img, (0, 0), (size[0] - 1, size[1] - 1), (255, 0, 0), 2)
            flag = False
        # add text label
        text_bg = np.zeros([20, g_img.shape[1], g_img.shape[2]], dtype='uint8')
        pid = pids[i + 1]
        text_bg = cv2.putText(text_bg, str(pid), (0, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        g_img = np.concatenate([g_img, text_bg], 0)
        if i == 0:
            rank1_success_marks = flag
        img_res.append(g_img)
    img_res = np.concatenate(img_res, 1)
    img_res = np.transpose(img_res, (2, 0, 1))
    img_res = torch.tensor(img_res, dtype=torch.uint8)
    #img_res = torch.unsqueeze(img_res,dim=0)
    return img_res, rank1_success_marks