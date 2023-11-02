from typing import List, Set, Dict, Tuple
import torch
from torchmetrics.classification import MulticlassAccuracy, accuracy
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--ind_logit_path", type=str, default=None, help="A saved torch tensor representing the logits of indomain evaluation dataset"
    )
    parser.add_argument(
        "--ood_logit_path", type=str, default=None, help="A saved torch tensor representing the logits of out-of-domain evaluation dataset"
    )
    parser.add_argument(
            "--ind_label_path", type=str, default=None, help="A saved torch tensor representing the labels of indomain evaluation dataset"
        )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    logits_ind = torch.load(args.ind_logit_path , map_location=torch.device('cpu'))
    ind = torch.Tensor([0]*logits_ind[0].shape[0])

    logits_ood = torch.load(args.ood_logit_path , map_location=torch.device('cpu'))
    ood = torch.Tensor([1]*logits_ind[0].shape[0])

    logits  = torch.cat((logits_ind, logits_ood ), axis=0) 
    is_ood = torch.cat((ind, ood), axis=-1)

    fpr, detection_err, auroc, aupr = calc_metrics(logits, is_ood)




def calc_metrics(logits, is_ood, echo=True):

    is_ind = torch.where(is_ood>0,0, 1 )
    softmax_max = torch.max(F.softmax(logits, dim=1),dim= 1)[0]

    fpr_in_tpr_95 = get_fpr_in_tpr_95(is_ind, softmax_max)
    detection_error = get_detection_error(is_ind, softmax_max)
    auroc = metrics.roc_auc_score(is_ind, softmax_max)

    precision, recall, thresholds = metrics.precision_recall_curve(is_ind, softmax_max)
    aupr_in = metrics.auc(recall, precision)

    if echo:
        print(f"fpr_in_tpr_95: {fpr_in_tpr_95}, minimum detection error: {detection_error.item()}, AUROC: {auroc}, AUPR In: {aupr_in}")
    return fpr_in_tpr_95, detection_error.item(), auroc, aupr_in

def calc_metrics_oracle(logits, is_ood, echo=True):

    is_ind = torch.where(is_ood>0,0, 1 )
    softmax_max = logits

    fpr_in_tpr_95 = get_fpr_in_tpr_95(is_ind, softmax_max)
    detection_error = get_detection_error(is_ind, softmax_max)
    auroc = metrics.roc_auc_score(is_ind, softmax_max)

    precision, recall, thresholds = metrics.precision_recall_curve(is_ind, softmax_max)
    aupr_in = metrics.auc(recall, precision)

    if echo:
        print(f"fpr_in_tpr_95: {fpr_in_tpr_95}, minimum detection error: {detection_error.item()}, AUROC: {auroc}, AUPR In: {aupr_in}")
    return fpr_in_tpr_95, detection_error.item(), auroc, aupr_in

def get_fpr_in_tpr_95(is_ind, softmax_max):
    fpr, tpr, thresholds = metrics.roc_curve(is_ind, softmax_max)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_in_tpr_95 = fpr[idx_tpr_95]
    return fpr_in_tpr_95




def get_detection_error(is_ind, softmax_max):
    sorted_indices = torch.argsort(softmax_max, descending=True)

    sorted_is_ind = is_ind[sorted_indices]
    sorted_softmax_max = softmax_max[sorted_indices]

    num_positives = sorted_is_ind.sum()
    num_negatives = len(sorted_is_ind) - num_positives

    cum_positives = sorted_is_ind.cumsum(dim=0)
    cum_negatives = torch.arange(1, len(sorted_is_ind) + 1) - cum_positives

    precision = cum_positives.float() / torch.arange(1, len(sorted_is_ind)+1).float()
    recall = cum_positives.float() / num_positives

    fpr = cum_negatives.float() / (num_negatives - cum_positives).float()
    accuracy = (cum_positives + (num_negatives - cum_negatives)).float() / len(sorted_is_ind)

    detection_error = 1 - accuracy

    return detection_error.min()