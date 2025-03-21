from torch import nn
import numpy as np
import torch
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)


def remove_nan_label(pred,truth):
    nan = torch.isnan(truth)
    truth = truth[~nan]
    pred = pred[~nan]

    return pred,truth

def roc_auc(pred,truth):
    return roc_auc_score(truth,pred)

def rmse(pred,truth):
    return nn.functional.mse_loss(pred,truth)**0.5

def mae(pred,truth):
    return mean_absolute_error(truth,pred)

func_dict={'relu':nn.ReLU(),
           'sigmoid':nn.Sigmoid(),
           'mse':nn.MSELoss(),
           'rmse':rmse,
           'mae':mae,
           'crossentropy':nn.CrossEntropyLoss(),
           'bce':nn.BCEWithLogitsLoss(),
           'auc':roc_auc,
           }

def get_func(fn_name):
    fn_name = fn_name.lower()
    return func_dict[fn_name]
