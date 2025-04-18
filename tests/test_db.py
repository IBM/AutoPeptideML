import os.path as osp

import numpy as np

from autopeptideml.db import Database
from autopeptideml.pipeline import Pipeline, CanonicalFilter


def test_database():
    dir_path = osp.abspath(osp.dirname(__file__))
    path = osp.join(dir_path, 'sample', 'example.csv')
    db = Database(path, feat_fields=['sequence'],
                  pipe=Pipeline([CanonicalFilter()]))
    assert len(db) == 500
    path2 = osp.join(dir_path, 'sample', 'example2.csv')
    db2 = Database(path2, feat_fields=['sequence'],
                   pipe=Pipeline([CanonicalFilter()]),
                   label_field='Y')
    db2.df['Y'] = 1
    db2.add_negatives(db, columns_to_exclude=['Allergen', 'Toxic'])
    labels, counts = np.unique(db2.df.Y, return_counts=True)
    assert labels.tolist() == [0, 1]
    assert counts.tolist() == [272, 300]
