import torch
import numpy as np
import cv2
import torch.nn.functional as F
from medpy import metric
from scipy.spatial.distance import directed_hausdorff

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    if type(x) is list:
        x = torch.tensor(np.array(x))
        y = torch.tensor(np.array(y))
        if x.min() < 0:
            x = m(x)
    else:
        x, y = x, y
        if x.min() < 0:
            x = m(x)
    return x, y


def iou(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3]) - intersection
    return ((intersection + eps) / (union + eps)).cpu().numpy()


def dice(pr, gt, eps=1e-7, threshold = 0.5):
    # pr = F.upsample_nearest(pr, size=(112,112))
    # gt = F.upsample_nearest(gt, size=(112,112))
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3])
    return ((2. * intersection +eps) / (union + eps)).cpu().numpy()


def dice_1(pr, gt, eps=1e-7, threshold = 0.5):
    # pr = F.upsample_nearest(pr, size=(112,112))
    # gt = F.upsample_nearest(gt, size=(112,112))
    pr_, gt_ = _list_tensor(pr[:,:1,:,:], gt[:,:1,:,:])
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3])
    return ((2. * intersection +eps) / (union + eps)).cpu().numpy()

def dice_2(pr, gt, eps=1e-7, threshold = 0.5):
    # pr = F.upsample_nearest(pr, size=(112,112))
    # gt = F.upsample_nearest(gt, size=(112,112))
    pr_, gt_ = _list_tensor(pr[:,1:2,:,:], gt[:,1:2,:,:])
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3])
    return ((2. * intersection +eps) / (union + eps)).cpu().numpy()

def dice_3(pr, gt, eps=1e-7, threshold = 0.5):
    # pr = F.upsample_nearest(pr, size=(112,112))
    # gt = F.upsample_nearest(gt, size=(112,112))
    pr_, gt_ = _list_tensor(pr[:,2:3,:,:], gt[:,2:3,:,:])
    pr_ = _threshold(pr_, threshold=threshold)
    gt_ = _threshold(gt_, threshold=threshold)
    intersection = torch.sum(gt_ * pr_,dim=[1,2,3])
    union = torch.sum(gt_,dim=[1,2,3]) + torch.sum(pr_,dim=[1,2,3])
    return ((2. * intersection +eps) / (union + eps)).cpu().numpy()

def hd(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr, gt)
    pr_ = _threshold(pr_, threshold=threshold).cpu().bool()
    gt_ = _threshold(gt_, threshold=threshold).cpu().bool()
    pred = np.array(pr_)
    gt = np.array(gt_)

    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        return hd95
    else:
        return 100

def hd_1(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr[:,:1,:,:], gt[:,:1,:,:])
    pr_ = _threshold(pr_, threshold=threshold).cpu().bool()
    gt_ = _threshold(gt_, threshold=threshold).cpu().bool()
    pred = np.array(pr_)
    gt = np.array(gt_)

    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        return hd95
    else:
        return 100

def hd_2(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr[:,1:2,:,:], gt[:,1:2,:,:])
    pr_ = _threshold(pr_, threshold=threshold).cpu().bool()
    gt_ = _threshold(gt_, threshold=threshold).cpu().bool()
    pred = np.array(pr_)
    gt = np.array(gt_)

    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        return hd95
    else:
        return 100

def hd_3(pr, gt, eps=1e-7, threshold = 0.5):
    pr_, gt_ = _list_tensor(pr[:,2:3,:,:], gt[:,2:3,:,:])
    pr_ = _threshold(pr_, threshold=threshold).cpu().bool()
    gt_ = _threshold(gt_, threshold=threshold).cpu().bool()
    pred = np.array(pr_)
    gt = np.array(gt_)

    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        return hd95
    else:
        return 100


import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def calculate_metrics(predictions, labels):
    # Flatten predictions and labels
    predictions_flat = predictions.reshape(-1)
    labels_flat = labels.reshape(-1)
    # Convert predictions to binary (0 or 1)
    predictions_binary = np.where(predictions_flat > 0.5, 1, 0)
    # True positives, false positives, true negatives, false negatives
    tp = np.sum(np.logical_and(predictions_binary == 1, labels_flat == 1))
    fp = np.sum(np.logical_and(predictions_binary == 1, labels_flat == 0))
    tn = np.sum(np.logical_and(predictions_binary == 0, labels_flat == 0))
    fn = np.sum(np.logical_and(predictions_binary == 0, labels_flat == 1))
    # Sensitivity (Recall)
    sensitivity = tp / (tp + fn)
    # Specificity
    specificity = tn / (tn + fp)
    # AUC
    auc = roc_auc_score(labels_flat, predictions_flat)
    # AUPR
    aupr = average_precision_score(labels_flat, predictions_flat)
    return sensitivity, specificity, auc, aupr

def metric_sim(pr, gt, eps=1e-7, threshold=0.5, metric='sen'):
    pr_, gt_ = _list_tensor(pr, gt)
    predictions = np.array(pr_.cpu())
    labels = np.array(gt_.cpu())
    predictions_flat = predictions.reshape(-1)
    labels_flat = labels.reshape(-1)
    # Convert predictions to binary (0 or 1)
    predictions_binary = np.where(predictions_flat >= threshold, 1, 0)
    # True positives, false positives, true negatives, false negatives
    tp = np.sum(np.logical_and(predictions_binary == 1, labels_flat == 1))
    fp = np.sum(np.logical_and(predictions_binary == 1, labels_flat == 0))
    tn = np.sum(np.logical_and(predictions_binary == 0, labels_flat == 0))
    fn = np.sum(np.logical_and(predictions_binary == 0, labels_flat == 1))
    
    if metric=='sen':
    # Sensitivity (Recall)
        sensitivity = tp / (tp + fn)
        return sensitivity
    elif metric == 'spec':
        specificity = tn / (tn + fp)
        return specificity
    elif metric == 'auc':
        auc = roc_auc_score(labels_flat, predictions_flat)
        return auc
    elif metric == 'aupr':
        aupr = average_precision_score(labels_flat, predictions_flat)
        return aupr

    return sensitivity



def sen_1(pr, gt, eps=1e-7, threshold=0.5):
    pr, gt = pr[:,0:1,:,:], gt[:,0:1,:,:]
    out = metric_sim(pr, gt, threshold, metric='sen')
    return out
def sen_2(pr, gt, eps=1e-7, threshold=0.5):
    pr, gt = pr[:,1:2,:,:], gt[:,1:2,:,:]
    out = metric_sim(pr, gt, threshold, metric='sen')
    return out
def sen_3(pr, gt, eps=1e-7, threshold=0.5):
    pr, gt = pr[:,2:3,:,:], gt[:,2:3,:,:]
    out = metric_sim(pr, gt, threshold, metric='sen')
    return out


def spec_1(pr, gt, eps=1e-7, threshold=0.5):
    pr, gt = pr[:,0:1,:,:], gt[:,0:1,:,:]
    out = metric_sim(pr, gt, threshold, metric='spec')
    return out
def spec_2(pr, gt, eps=1e-7, threshold=0.5):
    pr, gt = pr[:,1:2,:,:], gt[:,1:2,:,:]
    out = metric_sim(pr, gt, threshold, metric='spec')
    return out
def spec_3(pr, gt, eps=1e-7, threshold=0.5):
    pr, gt = pr[:,2:3,:,:], gt[:,2:3,:,:]
    out = metric_sim(pr, gt, threshold, metric='spec')
    return out

def auc_1(pr, gt, eps=1e-7, threshold=0.5):
    pr, gt = pr[:,0:1,:,:], gt[:,0:1,:,:]
    out = metric_sim(pr, gt, threshold, metric='auc')
    return out
def auc_2(pr, gt, eps=1e-7, threshold=0.5):
    pr, gt = pr[:,1:2,:,:], gt[:,1:2,:,:]
    out = metric_sim(pr, gt, threshold, metric='auc')
    return out
def auc_3(pr, gt, eps=1e-7, threshold=0.5):
    pr, gt = pr[:,2:3,:,:], gt[:,2:3,:,:]
    out = metric_sim(pr, gt, threshold, metric='auc')
    return out

def aupr_1(pr, gt, eps=1e-7, threshold=0.5):
    pr, gt = pr[:,0:1,:,:], gt[:,0:1,:,:]
    out = metric_sim(pr, gt, threshold, metric='aupr')
    return out
def aupr_2(pr, gt, eps=1e-7, threshold=0.5):
    pr, gt = pr[:,1:2,:,:], gt[:,1:2,:,:]
    out = metric_sim(pr, gt, threshold, metric='aupr')
    return out
def aupr_3(pr, gt, eps=1e-7, threshold=0.5):
    pr, gt = pr[:,2:3,:,:], gt[:,2:3,:,:]
    out = metric_sim(pr, gt, threshold, metric='aupr')
    return out




def SegMetrics(pred, label, metrics):
    metric_list = []  
    if isinstance(metrics, str):
        metrics = [metrics, ]
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'iou':
            metric_list.append(np.mean(iou(pred, label)))
        elif metric == 'dice':
            metric_list.append(np.mean(dice(pred, label)))
        elif metric == 'dice_1':
             metric_list.append(np.mean(dice_1(pred, label)))
        elif metric == 'dice_2':
             metric_list.append(np.mean(dice_2(pred, label)))
        elif metric == 'dice_3':
             metric_list.append(np.mean(dice_3(pred, label)))
        elif metric == 'hd':
            metric_list.append(np.mean(hd(pred, label)))
        elif metric == 'hd_1':
            metric_list.append(np.mean(hd_1(pred, label)))
        elif metric == 'hd_2':
            metric_list.append(np.mean(hd_2(pred, label)))
        elif metric == 'hd_3':
            metric_list.append(np.mean(hd_3(pred, label)))
        elif metric == 'sen_1':
            metric_list.append(np.mean(sen_1(pred, label)))
        elif metric == 'sen_2':
            metric_list.append(np.mean(sen_2(pred, label)))
        elif metric == 'sen_3':
            metric_list.append(np.mean(sen_3(pred, label)))
        elif metric == 'spec_1':
            metric_list.append(np.mean(spec_1(pred, label)))
        elif metric == 'spec_2':
            metric_list.append(np.mean(spec_2(pred, label)))
        elif metric == 'spec_3':
            metric_list.append(np.mean(spec_3(pred, label)))
        elif metric == 'auc_1':
            metric_list.append(np.mean(auc_1(pred, label)))
        elif metric == 'auc_2':
            metric_list.append(np.mean(auc_2(pred, label)))
        elif metric == 'auc_3':
            metric_list.append(np.mean(auc_3(pred, label)))
        elif metric == 'aupr_1':
            metric_list.append(np.mean(aupr_1(pred, label)))
        elif metric == 'aupr_2':
            metric_list.append(np.mean(aupr_2(pred, label)))
        elif metric == 'aupr_3':
            metric_list.append(np.mean(aupr_3(pred, label)))
        else:
            raise ValueError('metric %s not recognized' % metric)
    if pred is not None:
        metric = np.array(metric_list)
    else:
        raise ValueError('metric mistakes in calculations')
    return metric