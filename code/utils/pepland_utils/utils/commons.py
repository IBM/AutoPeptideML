#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/biology_llm_platform/mlm/pepland/utils/commons.py
# Project: /home/richard/projects/biology_llm_platform/mlm/pepland/utils
# Created Date: Thursday, November 28th 2024, 10:42:24 am
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Tue Dec 03 2024
# Modified By: Ruochi Zhang
# -----
# Copyright (c) 2024 Bodkin World Domination Enterprises
#
# MIT License
#
# Copyright (c) 2024 Ruochi Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----
###

import sys
import os
import mlflow
from typing import List, Union
import torch
import torch.nn as nn
import dgl
from rdkit import Chem


def load_model(model_path):
    """ Load the model from the model directory
        args:
            model_path: str, the path to the model directory
        return:
            model: torch.nn.Module
    """
    # from mlm.pepland.model.model import PharmHGT
    sys.path.insert(0, os.path.join(model_path, "code"))
    print("loading model from : {}".format(model_path))
    model = mlflow.pytorch.load_model(model_path, map_location="cpu")
    return model


def split_batch(bg, ntype, field, device):
    hidden = bg.nodes[ntype].data[field]
    node_size = bg.batch_num_nodes(ntype)
    start_index = torch.cat(
        [torch.tensor([0], device=device),
         torch.cumsum(node_size, 0)[:-1]])
    max_num_node = max(node_size)
    # padding
    hidden_lst = []
    for i in range(bg.batch_size):
        start, size = start_index[i], node_size[i]
        assert size != 0, size
        cur_hidden = hidden.narrow(0, start, size)
        cur_hidden = torch.nn.ZeroPad2d(
            (0, 0, 0, max_num_node - cur_hidden.shape[0]))(cur_hidden)
        hidden_lst.append(cur_hidden.unsqueeze(0))
    hidden_lst = torch.cat(hidden_lst, 0)
    return hidden_lst


class Permute(nn.Module):

    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, x):
        return torch.permute(x, (0, 2, 1))


class Squeeze(nn.Module):

    def __init__(self, dim):
        super(Squeeze, self).__init__()

        self.dim = dim

    def forward(self, x):
        return torch.squeeze(x, dim=self.dim)


def to_canonical_smiles(smiles: Union[str, List[str]]) -> List[str]:

    if isinstance(smiles, str):
        smiles = [smiles]

    canonical_smiles = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            canonical_smiles.append(Chem.MolToSmiles(mol))
        else:
            canonical_smiles.append(None)

    return canonical_smiles
