import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import random
import mlflow
import sys
from torch.backends import cudnn


def get_device(cfg):
    device = torch.device(
        "cuda:{}".format(cfg.train.device_ids[0]) if torch.cuda.is_available()
        and len(cfg.train.device_ids) > 0 else "cpu")
    return device

def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel,
                           nn.parallel.DistributedDataParallel)
    
def load_weights(model, best_model_path, device):

    best_model_path = os.path.join(best_model_path, "data/model.pth")

    if is_parallel(model):
        model = model.module

    model_dict = model.state_dict()

    best_state_dict = {
        k.replace("module.", ""): v
        for (k, v) in list(
            torch.load(best_model_path,
                       map_location="cpu").state_dict().items())
    }

    model_dict.update(best_state_dict)
    model.load_state_dict(model_dict)

    model.to(device)

    return model

def fix_random_seed(random_seed, cuda_deterministic=True):
    # fix random seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
        
def load_model_masking(model_path, device):
    root_dir = os.getcwd()
    model_path = os.path.join(root_dir, model_path)
    print("loading model from : {}".format(model_path))
    print(os.path.join(model_path, 'model'))
    sys.path.append(os.path.join(os.path.join(model_path, 'model'), "code"))

    model = mlflow.pytorch.load_model(os.path.join(model_path, 'model'), map_location="cpu").to(device)
    linear_pred_atoms = mlflow.pytorch.load_model(os.path.join(model_path,'linear_pred_atoms'), map_location="cpu").to(device)
    linear_pred_bonds = mlflow.pytorch.load_model(os.path.join(model_path,'linear_pred_bonds'), map_location="cpu").to(device)
    # model.eval()
    # linear_pred_atoms.eval()
    # linear_pred_bonds.eval()
    return model, linear_pred_atoms, linear_pred_bonds

def load_model_contextpred(model_path, device):
    root_dir = os.getcwd()
    model_path = os.path.join(root_dir, model_path)
    print("loading model from : {}".format(model_path))
    model_substruct = mlflow.pytorch.load_model(os.path.join(model_path, 'model_substruct'), map_location="cpu").to(device)
    model_context = mlflow.pytorch.load_model(os.path.join(model_path,'model_context'), map_location="cpu").to(device)
    model_substruct.eval()
    model_context.eval()
    return model_substruct, model_context