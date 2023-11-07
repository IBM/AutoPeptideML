import math
import time
from typing import Tuple, Dict, List

import networkx as nx
import numpy as np
import pandas as pd

from .partitioning.mmseqs_utils import generate_edges_mmseqs
from .partitioning.needle_utils import generate_edges_needle


def _assing_components(
        components: list,
        g: nx.Graph,
        labels: list,
        PARTITION_TARGET: Dict[str, int],
        PARTITION_LABELS: List[Dict[str, int]]
) -> List[Dict[int, Dict]]:

    PARTITIONS = {
        0: {'samples': [], 'labels': {l: 0 for l in PARTITION_LABELS[0]}}, 
        1: {'samples': [], 'labels': {l: 0 for l in PARTITION_LABELS[1]}}
    }
    component_inds = np.argsort(list(map(len, components)))

    for ind in component_inds:
        component = components[ind]
        tmp_c_labels = np.array([labels[c] for c in component])
        component_labels = {l: np.sum(tmp_c_labels == l) for l in np.unique(labels)}

        if len(PARTITIONS[1]['samples']) + len(component) < PARTITION_TARGET['test']:
            continue_flag = False
            for l in PARTITION_LABELS[1]:
                if PARTITIONS[1]['labels'][l] + component_labels[l] > PARTITION_LABELS[1][l]:
                    continue_flag = True

            if not continue_flag:
                PARTITIONS[1]['samples'].extend(component)
                for l in PARTITION_LABELS[1]:
                    PARTITIONS[1]['labels'][l] += component_labels[l]
            else:
                PARTITIONS[0]['samples'].extend(component)
                for l in PARTITION_LABELS[0]:
                    PARTITIONS[0]['labels'][l] += component_labels[l]
        else:
            PARTITIONS[0]['samples'].extend(component)
            for l in PARTITION_LABELS[0]:
                PARTITIONS[0]['labels'][l] += component_labels[l]
    

    return PARTITIONS

def _pandas2graph(df: pd.DataFrame) -> Tuple[nx.Graph, list, list]:
    full_graph = nx.Graph()
    sequences = df['sequence'].tolist()
    full_graph.add_nodes_from(df.index.tolist())
    labels = df['Y'].tolist()
    return full_graph, sequences, labels

def make_graphs_from_sequences(
        df: pd.DataFrame,
        verbose: int,
        alignment: str,
        outputdir: str,
        denominator: str,
        threads: int,
        threshold: float,
        **kwargs
    ):
    full_graph, seqs, labels = _pandas2graph(df, **kwargs)
    start = time.perf_counter()

    if alignment == 'mmseqs':
        generate_edges_mmseqs(
            full_graph,
            sequences=seqs,
            verbose=verbose,
            outputdir=outputdir,
            denominator=denominator,
            use_prefilter=False,
            threads=threads,
            threshold=threshold
        )
    elif alignment == 'mmseqs+prefilter':
        generate_edges_mmseqs(
            full_graph,
            sequences=seqs,
            verbose=verbose,
            outputdir=outputdir,
            denominator=denominator,
            use_prefilter=True,
            threads=threads,
            threshold=threshold
        )
    elif alignment == 'needle':
        generate_edges_needle(
            full_graph,
            sequences=seqs,
            verbose=verbose,
            outputdir=outputdir,
            denominator=denominator,
            threads=threads,
            threshold=threshold
        )
    else: 
        raise NotImplementedError(
            f'Alignment method: {alignment} is currently not supported',
            'Please use one of the following supported methods: `mmseqs`,',
            '`mmseqs+prefilter`, `needle`.')

    end = time.perf_counter()
    elapsed = end - start
    if verbose > 1:
        print(f'Graph built in {elapsed:0.2f} s')
    return full_graph, seqs, labels

def train_test(
    g: nx.Graph,
    ids: list,
    sequences: list,
    labels: list,
    test_size: float=0.2,
    threshold: float=0.3,
    verbose: int=2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    edges = list(g.edges(data=True))
    nodes = g.nodes(data=True)
    nodes = {node[0]: node[1] for node in nodes}
    edges_to_remove = [edge for edge in edges 
                       if edge[2]['metric'] <= threshold]
    g.remove_edges_from(edges_to_remove)

    PARTITION_TARGET = {
        'train': math.floor(len(sequences) * (1 - test_size)),
        'test': math.ceil(len(sequences) * test_size)
    }
    PARTITION_LABELS = [
        {l: math.floor((len(labels) / len(np.unique(labels))) * (1 - test_size)) for l in np.unique(labels)},
        {l: math.ceil((len(labels) / len(np.unique(labels)))  * (test_size)) for l in np.unique(labels)},
    ]
    components = []
    components = list(nx.connected_components(g)) 
    division = 0
    while not (test_size * 0.9 < division < test_size * 1.1):
        PARTITIONS = _assing_components(components, g, labels, PARTITION_TARGET, PARTITION_LABELS)
        division = (len(PARTITIONS[1]['samples']) / 
            (len(PARTITIONS[0]['samples']) + len(PARTITIONS[1]['samples'])))
        if not test_size * 0.9 < division < test_size * 1.1:
            raise RuntimeError(f'It is impossible to partition the dataset at current threshold: {threshold}',
                               f' and test size: {test_size}. Please modify either of the parameters.')

    data = []
    for label, partition in PARTITIONS.items():
        for node in partition['samples']:
            data.append({
                'id': ids[node],
                'sequence': sequences[node],
                'Y': int(labels[node]),
                'partition': label
            })
    df = pd.DataFrame(data)
    train_df = df[df.partition == 0].copy()
    test_df = df[df.partition == 1].copy()
    del df
    train_df.drop(columns='partition', inplace=True)
    test_df.drop(columns='partition', inplace=True)
    if verbose > 1:
        print(f'Test label balance: {(len(test_df[test_df.Y == 0]) / len(test_df)):.2f}')
    return train_df, test_df
