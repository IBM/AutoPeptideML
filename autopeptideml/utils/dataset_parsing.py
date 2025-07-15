import os.path as osp

from os import listdir

import pandas as pd


def read_file(path: str) -> pd.DataFrame:
    if path.endswith('.fasta'):
        return _read_fasta(path)
    elif path.endswith('.smi'):
        return _read_smi(path)
    elif path.endswith('.csv') or path.endswith('.tsv'):
        return _read_csv(path)
    else:
        raise RuntimeError(f"File format: {path.split('.')[-1]} is not supported.",
                           "Supported formats: `.smi`, `.fasta`, `.csv`, or `.tsv`")


def read_data(path: str) -> pd.DataFrame:
    df = pd.DataFrame()
    if osp.isdir(path):
        for filepath in listdir(path):
            filepath = osp.join(path, filepath)
            df = pd.concat([df, read_data(path)])
    else:
        df = pd.concat([df, read_file(path)])
    return df


def _read_smi(path: str) -> pd.DataFrame:
    text = [r.strip() for r in open(path).readlines()]
    return pd.DataFrame({'SMILES': text})


def _read_fasta(path: str) -> pd.DataFrame:
    out = []
    text = [r.strip() for r in open(path).readlines()]
    for idx, line in enumerate(text):
        if line.startswith(">"):
            if idx > 0:
                out.append(entry)

            entry = {"header": line, "sequence": ""}
        else:
            entry["sequence"] += line
    return pd.DataFrame(out)


def _read_csv(path: str) -> pd.DataFrame:
    if path.endswith('.csv'):
        return pd.read_csv(path, sep=',')
    elif path.endswith(".tsv"):
        return pd.read_csv(path, sep='\t')
