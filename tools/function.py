import os
from collections import OrderedDict

import numpy as np
import torch
from easydict import EasyDict

from tools.utils import may_mkdirs
from clipS.model import build_model


def ratio2weight(targets, ratio):
    ratio = torch.from_numpy(ratio).type_as(targets)
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights[targets > 1] = 0.0

    return weights

def get_model_log_path(root_path, model_name):
    multi_attr_model_dir = os.path.join(root_path, model_name, 'img_model')
    may_mkdirs(multi_attr_model_dir)

    multi_attr_log_dir = os.path.join(root_path, model_name, 'log')
    may_mkdirs(multi_attr_log_dir)

    return multi_attr_model_dir, multi_attr_log_dir

class LogVisual:

    def __init__(self, args):
        self.args = vars(args)
        self.train_loss = []
        self.val_loss = []

        self.ap = []
        self.map = []
        self.acc = []
        self.prec = []
        self.recall = []
        self.f1 = []

        self.error_num = []
        self.fn_num = []
        self.fp_num = []

        self.save = False

    def append(self, **kwargs):
        self.save = False

        if 'result' in kwargs:
            self.ap.append(kwargs['result']['label_acc'])
            self.map.append(np.mean(kwargs['result']['label_acc']))
            self.acc.append(np.mean(kwargs['result']['instance_acc']))
            self.prec.append(np.mean(kwargs['result']['instance_precision']))
            self.recall.append(np.mean(kwargs['result']['instance_recall']))
            self.f1.append(np.mean(kwargs['result']['floatance_F1']))

            self.error_num.append(kwargs['result']['error_num'])
            self.fn_num.append(kwargs['result']['fn_num'])
            self.fp_num.append(kwargs['result']['fp_num'])

        if 'train_loss' in kwargs:
            self.train_loss.append(kwargs['train_loss'])
        if 'val_loss' in kwargs:
            self.val_loss.append(kwargs['val_loss'])


def get_pkl_rootpath(dataset):
    root = os.path.join("./dataset", f"{dataset}")
    data_path = os.path.join(root, 'dataset.pkl')

    return data_path


def get_pedestrian_metrics(gt_label, preds_probs, threshold=0.45):#0.45,0.5
    pred_label = preds_probs > threshold
    # pred_label = []
    # for i in range(preds_probs.shape[0]):
    #     pred_label.append(group_check(preds_probs[i]))
    # pred_label = np.array(pred_label)

    eps = 1e-20
    result = EasyDict()

    ###############################
    # label metrics
    # TP + FN
    gt_pos = np.sum((gt_label == 1), axis=0).astype(float)
    # TN + FP
    gt_neg = np.sum((gt_label == 0), axis=0).astype(float)
    # TP
    true_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=0).astype(float)
    # TN
    true_neg = np.sum((gt_label == 0) * (pred_label == 0), axis=0).astype(float)
    # FP
    false_pos = np.sum(((gt_label == 0) * (pred_label == 1)), axis=0).astype(float)
    # FN
    false_neg = np.sum(((gt_label == 1) * (pred_label == 0)), axis=0).astype(float)

    label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
    label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
    # mean accuracy
    label_ma = (label_pos_recall + label_neg_recall) / 2

    result.label_pos_recall = label_pos_recall
    result.label_neg_recall = label_neg_recall
    result.label_prec = true_pos / (true_pos + false_pos + eps)
    result.label_acc = true_pos / (true_pos + false_pos + false_neg + eps)
    result.label_f1 = 2 * result.label_prec * result.label_pos_recall / (
            result.label_prec + result.label_pos_recall + eps)

    result.label_ma = label_ma
    result.ma = np.mean(label_ma)

    ################
    # instance metrics
    gt_pos = np.sum((gt_label == 1), axis=1).astype(float)
    true_pos = np.sum((pred_label == 1), axis=1).astype(float)
    # true positive
    intersect_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=1).astype(float)
    # IOU
    union_pos = np.sum(((gt_label == 1) + (pred_label == 1)), axis=1).astype(float)

    instance_acc = intersect_pos / (union_pos + eps)
    instance_prec = intersect_pos / (true_pos + eps)
    instance_recall = intersect_pos / (gt_pos + eps)
    instance_f1 = 2 * instance_prec * instance_recall / (instance_prec + instance_recall + eps)

    instance_acc = np.mean(instance_acc)
    instance_prec = np.mean(instance_prec)
    instance_recall = np.mean(instance_recall)
    instance_f1 = np.mean(instance_f1)

    result.instance_acc = instance_acc
    result.instance_prec = instance_prec
    result.instance_recall = instance_recall
    result.instance_f1 = instance_f1

    result.error_num, result.fn_num, result.fp_num = false_pos + false_neg, false_neg, false_pos

    return result,pred_label

def get_pedestrian_metrics0(gt_label, preds_probs,threshold=0.45):# threshold=0.7):
    #pdb.set_trace() 
    # print("-------------------------------------------------") 
    # print(preds_probs) 
    #pred_label = preds_probs > threshold
    pred_label = preds_probs
    # print(gt_label)
    # print(pred_label)
    # print("-------------------------------------------------")

    eps = 1e-20
    result = EasyDict()


    ###############################
    # label metrics
    # TP + FN
    gt_pos = np.sum((gt_label == 1), axis=0).astype(float)
    # TN + FP
    gt_neg = np.sum((gt_label == 0), axis=0).astype(float)
    # TP
    true_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=0).astype(float)
    # TN
    true_neg = np.sum((gt_label == 0) * (pred_label == 0), axis=0).astype(float)
    # FP
    false_pos = np.sum(((gt_label == 0) * (pred_label == 1)), axis=0).astype(float)
    # FN
    false_neg = np.sum(((gt_label == 1) * (pred_label == 0)), axis=0).astype(float)

    label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
    label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
    # mean accuracy
    label_ma = (label_pos_recall + label_neg_recall) / 2

    result.label_pos_recall = label_pos_recall
    result.label_neg_recall = label_neg_recall
    result.label_prec = true_pos / (true_pos + false_pos + eps)
    result.label_acc = true_pos / (true_pos + false_pos + false_neg + eps)
    result.add_acc = (true_pos+true_neg) / (true_pos + false_pos + false_neg +true_neg+ eps)
    result.label_f1 = 2 * result.label_prec * result.label_pos_recall / (
            result.label_prec + result.label_pos_recall + eps)

    result.label_ma = label_ma
    result.ma = np.mean(label_ma)

    ################
    # instance metrics
    gt_pos = np.sum((gt_label == 1), axis=1).astype(float)
    true_pos = np.sum((pred_label == 1), axis=1).astype(float)
    # true positive
    intersect_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=1).astype(float)
    # IOU
    union_pos = np.sum(((gt_label == 1) + (pred_label == 1)), axis=1).astype(float)

    instance_acc = intersect_pos / (union_pos + eps) #TP/TP+FP   pre
    instance_prec = intersect_pos / (true_pos + eps) 
    instance_recall = intersect_pos / (gt_pos + eps)
    instance_f1 = 2 * instance_prec * instance_recall / (instance_prec + instance_recall + eps)

    instance_acc = np.mean(instance_acc)
    instance_prec = np.mean(instance_prec)
    instance_recall = np.mean(instance_recall)
    instance_f1 = np.mean(instance_f1)

    result.instance_acc = instance_acc
    result.instance_prec = instance_prec
    result.instance_recall = instance_recall
    result.instance_f1 = instance_f1

    result.error_num, result.fn_num, result.fp_num = false_pos + false_neg, false_neg, false_pos

    return result

def get_reload_weight(model_path, model,pth='ckpt_max.pth'):
    model_path = os.path.join(model_path, pth)
    load_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    if isinstance(load_dict, OrderedDict):
        pretrain_dict = load_dict
    else:
        pretrain_dict = load_dict['model_state_dicts']
        clip_pretrain_dict=load_dict['clip_model']
        print(f"best performance {load_dict['metric']} in epoch : {load_dict['epoch']}")

    model.load_state_dict(pretrain_dict, strict=True)
    clip_model=build_model(clip_pretrain_dict).cuda()
    # clip_model.load_state_dict(clip_pretrain_dict, strict=True)
    # group_arr=load_dict['group_arr']
    return model,clip_model#,group_arr

def group_check(atlb):
    atlb = torch.tensor(atlb)
    c_atlb=torch.zeros(len(atlb),dtype=torch.int64)#.shape[0]
    # for i in range(atlb.shape[0]):
    if atlb[34]>0.5:
        c_atlb[34] = 1#male
        
    ind_age= torch.argmax(atlb[30:34])#age
    c_atlb[30+ind_age] = 1
    
    if atlb[4]>0.5:
        c_atlb[4] = 1#hair
    
    ind_bag = torch.argmax(atlb[25:30])#bag
    if ind_bag == 3:
        c_atlb[28] = 1
    else:
        c_atlb[[25,26,27,29]] = torch.where(atlb[[25,26,27,29]] > 0.5, torch.tensor(1, dtype=torch.int64), torch.tensor(0, dtype=torch.int64)) 
    
    ind_head = torch.argmax(atlb[:4])#headwear
    if ind_head == 2:
        c_atlb[2] = 1
    else:
        c_atlb[[0,1,3]] = torch.where(atlb[[0,1,3]] > 0.5, torch.tensor(1, dtype=torch.int64), torch.tensor(0, dtype=torch.int64))
        
    ind_upstyle= torch.argmax(atlb[5:7])#casual/fomal upperwear
    c_atlb[5+ind_upstyle] = 1
    
    c_atlb[7:15] = torch.where(atlb[7:15] > 0.5, torch.tensor(1, dtype=torch.int64), torch.tensor(0, dtype=torch.int64)) #uppercloth,未做互斥检测
    
    ind_upstyle= torch.argmax(atlb[15:17])#casual/fomal lowerwear
    c_atlb[15+ind_upstyle] = 1
    
    ind_low= torch.argmax(atlb[17:21])#lowerwear
    c_atlb[17+ind_low] = 1
    
    ind_shoe= torch.argmax(atlb[21:25])#shoes
    c_atlb[21+ind_shoe] = 1
    return c_atlb