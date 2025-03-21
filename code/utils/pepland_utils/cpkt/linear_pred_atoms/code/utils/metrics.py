import traceback
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score
import torch
from collections import defaultdict



def cal_recall(y_pred, y_true, top):
    a_sort_idx = y_pred.argsort()
    b_sort_idx = y_true.argsort()
    
    recall = len(set(b_sort_idx[-top:].tolist()).intersection(a_sort_idx[-top:].tolist()))

    return recall


class MulticlassMetrics():
    def __init__(self):
        pass
    
    def __call__(self, pred, true):
        
        pred = torch.argmax(torch.softmax(pred, dim=-1), dim = -1).cpu().numpy()
        true = true.cpu().numpy()

        res_dict = {}

        report = classification_report(true, pred)
        
        
        pre_score = precision_score(true, pred, average="macro")
        rec_score = recall_score(true, pred, average="macro")
        try:
            auc_score = roc_auc_score(true, pred, average="macro")
        except:
            traceback.print_exc()
            auc_score = 0
            
        res_dict["auc_score"] = auc_score
        res_dict["precision_score"] = pre_score
        res_dict["recall_score"] = rec_score
        res_dict["classification_report"] = report

        return res_dict

            
class AffinityMetrics():
    def __init__(self, topK):
        self.topK = topK
        
    def __call__(self, pred, true):

        true = true.cpu().numpy()
        pred = pred.squeeze(1).cpu().numpy()       

        # print(pred)
        # print(true)
            
        # Get the predictions
        mse = (np.square(pred - true)).mean()
        pearson_corr = pearsonr(pred, true)[0]
        spearman_corr = spearmanr(pred, true)[0]

        recall = cal_recall(pred, true, self.topK)

        return {"mse": mse, "pearson": pearson_corr, "spearman": spearman_corr, "recall": recall}



class Metrics():
    def __init__(self, topK):
        
        self.affinity_metrics = AffinityMetrics(topK)
        self.multiclass_metrics = MulticlassMetrics()

    def __call__(self, pred_dict_list, true_dict_list):

        tasks = list(pred_dict_list[0].keys())

        res = {}

        for task in tasks:   
            tmp_pred_list = []
            tmp_true_list = []
            for pred_dict, true_dict in zip(pred_dict_list, true_dict_list):
                tmp_pred_list.append(pred_dict[task])
                tmp_true_list.append(true_dict[task])

            y_pred = torch.cat(tmp_pred_list, axis = 0)
            y_true = torch.cat(tmp_true_list, axis = 0)

            if task == "affinity":
                metrics = self.affinity_metrics(y_pred, y_true)

            res.update(metrics)

        return res
            
        
            
            
                
                
                
            

                
    