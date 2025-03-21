#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/projects/pepcorsser/models/core.py
# Project: /home/richard/projects/biology_llm_platform/mlm/pepland/model
# Created Date: Sunday, April 28th 2024, 12:03:00 am
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Wed Dec 04 2024
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
import os
import sys
from ..utils.commons import load_model, split_batch, Permute, Squeeze, to_canonical_smiles
from ..utils.process import Mol2HeteroGraph
import torch
import torch.nn as nn
import numpy as np
from typing import List, Union
import dgl
from collections import OrderedDict


class Node_GRU(nn.Module):
    """GRU for graph readout. Implemented with dgl graph"""

    def __init__(self, hid_dim=300, bidirectional=True):
        super(Node_GRU, self).__init__()
        self.hid_dim = hid_dim
        if bidirectional:
            self.direction = 2
        else:
            self.direction = 1

        self.atom_gru = nn.GRU(hid_dim,
                               hid_dim,
                               batch_first=True,
                               bidirectional=bidirectional)
        self.frag_gru = nn.GRU(hid_dim,
                               hid_dim,
                               batch_first=True,
                               bidirectional=bidirectional)

        factor = 2 if bidirectional else 1
        self.projection = nn.Linear(hid_dim * 2 * factor, hid_dim)

    def forward(self, atom_rep: torch.Tensor,
                frag_rep: torch.Tensor) -> torch.Tensor:

        atom_rep, _ = self.atom_gru(atom_rep)
        frag_rep, _ = self.frag_gru(frag_rep)

        atom_rep = atom_rep[:, -1, :]
        frag_rep = frag_rep[:, -1, :]

        embed = torch.cat([atom_rep, frag_rep], dim=1)

        embed = self.projection(embed)

        return embed


class PepLandFeatureExtractor(nn.Module):

    def __init__(self,
                 model_path,
                 pooling: Union[str, None] = 'avg',
                 freeze=True):
        """ Initialize the PepLandInference class
            args:
                model_path: str, the path to the model directory
                pooling: str, the pooling method, either 'max', 'avg', or 'gru'
                freeze: bool, whether to freeze the model
        """
        super(PepLandFeatureExtractor, self).__init__()

        self.model = load_model(model_path)

        # Remove layers containing "readout" and "out"
        for name, module in list(self.model.named_children()):
            if 'readout' in name or 'out' in name:
                delattr(self.model, name)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        if pooling == 'max':
            pooling_layer = nn.Sequential(Permute(),
                                          nn.AdaptiveMaxPool1d(output_size=1),
                                          Squeeze(dim=-1))
        elif pooling == 'avg':
            pooling_layer = nn.Sequential(Permute(),
                                          nn.AdaptiveAvgPool1d(output_size=1),
                                          Squeeze(dim=-1))
        elif pooling == "gru":
            pooling_layer = Node_GRU(hid_dim=300, bidirectional=True)
        else:
            pooling_layer = None
        self.pooling_layer = pooling_layer

        self.max_cache_size = 100000  # Set the maximum cache size

        # Initialize the cache as an OrderedDict
        # key: SMILES string, value: DGLHeteroGraph
        self._tokenize_cache = OrderedDict()

    def tokenize(self, input_smiles: List[str]) -> List:
        input_smiles = to_canonical_smiles(input_smiles)

        graphs = []
        for smi in input_smiles:
            if smi in self._tokenize_cache:
                # Move the recently accessed item to the end to mark it as recently used
                graph = self._tokenize_cache.pop(smi)
                self._tokenize_cache[smi] = graph
            else:
                try:
                    graph = Mol2HeteroGraph(smi)
                    # If cache is full, remove the least recently used item
                    if len(self._tokenize_cache) >= self.max_cache_size:
                        removed_smi, _ = self._tokenize_cache.popitem(
                            last=False)
                        # Optionally, log or print which SMILES was removed
                        # print(f"Cache full. Removed least recently used SMILES: {removed_smi}")
                    self._tokenize_cache[smi] = graph  # Add new item to cache
                except Exception as e:
                    raise ValueError(
                        f"Error processing SMILES string: {smi}. {e}")
            graphs.append(graph)
        return graphs

    def extract_atom_fragment_embedding(
            self, input_smiles: Union[List[str],
                                      dgl.DGLHeteroGraph]) -> torch.Tensor:
        """ Extract the atom and fragment embedding from the model
            args:
                input_smiles: List of SMILES strings or DGL graphs
            return:
                atom_embeds: torch.Tensor
                frag_embeds: torch.Tensor

            examples:
                input_smiles = ['CCO', 'CCN']
                atom_embeds.shape == [2, 30, 300]
                frag_embeds.shape == [2, 300]
        """

        if isinstance(input_smiles, str):
            input_smiles = [input_smiles]

        if not isinstance(input_smiles[0], dgl.DGLHeteroGraph):
            graphs = self.tokenize(input_smiles)
        else:
            graphs = input_smiles

        bg = dgl.batch(graphs).to(self.device)

        with torch.no_grad():
            atom_embed, frag_embed = self.model(bg)

        bg.nodes['a'].data['h'] = atom_embed
        bg.nodes['p'].data['h'] = frag_embed

        atom_embeds = split_batch(bg, 'a', 'h', self.device)
        frag_embeds = split_batch(bg, 'p', 'h', self.device)

        return atom_embeds, frag_embeds

    def forward(
            self,
            input_smiles: List[str],
            atom_index: Union[int, List[int], None] = None) -> torch.Tensor:
        """ Extract the peptide embedding from the model
            args:
                input_smiles: List of SMILES strings
                atom_index: if set, only return the atom embedding with the index
            return:
                pep_embeds: torch.Tensor

            examples:
                input_smiles = ['CCO', 'CCN']
                atom_index = 1
                pep_embeds.shape == [2, 1, 300]

                input_smiles = ['CCO', 'CCN']
                atom_index = [1, 2]
                pep_embeds.shape == [2, 2, 300]

                input_smiles = ['CCO', 'CCN']
                atom_index = None
                pep_embeds.shape == [2, 300]
        """
        atom_rep, frag_rep = self.extract_atom_fragment_embedding(input_smiles)

        # If atom_index is set, only return the atom embedding with the index
        if atom_index is not None:
            pep_embeds = atom_rep[:, atom_index]
        else:
            # If not set atom index, return the whole peptide embedding (atom + fragment)
            if self.pooling_layer is not None:
                if isinstance(self.pooling_layer, Node_GRU):
                    embed = self.pooling_layer(atom_rep, frag_rep)
                else:
                    embed = self.pooling_layer(
                        torch.cat([atom_rep, frag_rep], dim=1))
            else:
                embed = torch.cat([atom_rep, frag_rep], dim=1)

            pep_embeds = embed

        return pep_embeds

    @property
    def device(self):
        return next(self.parameters()).device


class PropertyPredictor(nn.Module):
    """ PropertyPredictor
        This model is used to predict the property of the peptide
        based on the peptide graph representation from pre-trained PepLand model.
    """

    def __init__(self,
                 model_path,
                 pooling="avg",
                 hidden_dims=[256, 128],
                 mlp_dropout=0.1):
        """ Initialize the PropertyPredictor class"""
        super(PropertyPredictor, self).__init__()

        self.feature_model = PepLandFeatureExtractor(model_path=model_path,
                                                     pooling=pooling)

        self.mlp = nn.Sequential()
        input_dim = 300
        for i, hidden_dim in enumerate(hidden_dims):
            self.mlp.add_module('linear_{}'.format(i),
                                nn.Linear(input_dim, hidden_dim))
            self.mlp.add_module('relu_{}'.format(i), nn.ReLU())
            self.mlp.add_module('dropout_{}'.format(i),
                                nn.Dropout(mlp_dropout))
            input_dim = hidden_dim

        self.mlp.add_module('output', nn.Linear(hidden_dim, 1))
        self.mlp.add_module('sigmoid', nn.Sigmoid())

    def tokenize(self, input_molecules: List[str]) -> List[dgl.DGLHeteroGraph]:
        """ Tokenize the input SMILES strings into DGLHeteroGraph
            args:
                input_molecules: List of SMILES strings
            return: List of dgl.DGLHeteroGraph
        """
        return self.feature_model.tokenize(input_molecules)

    def forward(
        self, input_molecules: Union[List[str], List[dgl.DGLHeteroGraph]]
    ) -> torch.Tensor:
        """ args:
            input_molecules: List of SMILES strings or dgl.DGLHeteroGraph
            return: torch.Tensor
        """

        graph_rep = self.feature_model(input_molecules)
        pred = self.mlp(graph_rep)

        return pred

    @property
    def device(self):
        return next(self.parameters()).device
