import os.path as osp

import numpy as np
import pandas as pd

from autopeptideml.db import add_negatives_from_db
from autopeptideml.pipeline import get_pipeline


def test_database():
    dir_path = osp.abspath(osp.dirname(__file__))
    path = osp.join(dir_path, 'sample', 'example.csv')
    df = pd.read_csv(path)
    path2 = osp.join(dir_path, 'sample', 'example2.csv')
    df2 = pd.read_csv(path2)
    assert len(df) == 500
    pipe = get_pipeline('to-smiles')
    df['smiles'] = pipe(df['sequence'])
    df2['smiles'] = pipe(df2['sequence'])
    df2['Y'] = 1
    df2 = add_negatives_from_db(df2, target_db=df, sequence_field='smiles',
                                activities_to_exclude=['Allergen'],
                                label_field='Y')
    labels, counts = np.unique(df2.Y, return_counts=True)
    assert labels.tolist() == [0, 1]
    assert counts.tolist() == [285, 300]
