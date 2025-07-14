import os
import os.path as osp

from multiprocessing import cpu_count, Pool
from typing import List, Tuple, Union

import pandas as pd

from ..pipeline import get_pipeline
from ..utils import discretizer


def __mw(mol: str):
    from rdkit.Chem import Descriptors, MolFromSmiles

    mol = MolFromSmiles(mol)
    return Descriptors.ExactMolWt(mol)


def _mw(input_str: List[str], n_jobs: int,
        verbose: bool = False) -> Tuple[List[float], int]:
    try:
        from rdkit.Chem import Descriptors, MolFromSmiles
    except ImportError:
        raise ImportError("Rdkit is required for this function",
                          "Please install: `pip install rdkit`")
    pipe = get_pipeline("to-smiles-fast")
    input_str = pipe(input_str, n_jobs=n_jobs, verbose=verbose)
    pool = Pool(processes=n_jobs)
    wts = pool.map(__mw, input_str)
    return wts, 50


def _length(input_str: List[str], n_jobs: int,
            verbose: bool) -> Tuple[List[int], int]:
    pipe = get_pipeline('to-sequences')
    input_str = pipe(input_str, n_jobs=n_jobs)
    lens = [len(i) for i in input_str]
    return lens, min(10, int(max(lens) - min(lens)))


TARGET_DBs = ['canonical', 'non-canonical', 'both']
MATCHING = {'mw': _mw, 'length': _length}


def add_negatives_from_db(
    df: pd.DataFrame,
    target_db: Union[str, pd.DataFrame],
    sequence_field: str,
    activities_to_exclude: List[str] = [],
    label_field: str = None,
    desired_ratio: float = 1.0,
    verbose: bool = True,
    sample_by: str = 'mw',
    n_jobs: int = cpu_count(),
    random_state: int = 1
) -> pd.DataFrame:
    if label_field is not None and label_field not in df.columns:
        raise ValueError(f"Label field {label_field} not in DataFrame.")
    if sequence_field not in df.columns:
        raise ValueError("Sequence field is not in DataFrame. Please check spelling.")
    if not (isinstance(target_db, str) or isinstance(target_db, pd.DataFrame)):
        raise ValueError("target_db has to be a string or a pd.DataFrame")
    if isinstance(target_db, str) and target_db not in TARGET_DBs:
        raise ValueError(f"target_db: {target_db} is not valid.",
                         f"Either provide a custom pd.DataFrame or use one of the prepared ones: {', '.join(TARGET_DBs)}")
    if sample_by not in MATCHING:
        raise ValueError(f"sample_by: {sample_by} is not valid.",
                         f"Please select one of the following options: {', '.join(list(MATCHING.keys()))}")
    if isinstance(activities_to_exclude, str):
        activities_to_exclude = [activities_to_exclude]
    db_dir = osp.join(osp.dirname(__file__), '..', 'data', 'dbs')
    if not osp.isdir(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    elif label_field is None:
        label_field = 'label'
        df[label_field] = 1

    if isinstance(target_db, pd.DataFrame):
        db = target_db

    elif target_db == 'canonical':
        path = osp.join(db_dir, 'canonical.csv')
        if not osp.exists(path):
            try:
                import gdown
            except ImportError:
                raise ImportError("This module requires gdown. Try: `pip install gdown`")
            print("Downloading negative database...")
            FILE_ID = "189VtkbQ2bVpQlAe2UMBSzt_O4F7EyBWl"
            gdown.download(id=FILE_ID, output=path, quiet=verbose)
        db = pd.read_csv(path)

    elif target_db == 'non-canonical':
        path = osp.join(db_dir, 'non-canonical.csv')
        if not osp.exists(path):
            try:
                import gdown
            except ImportError:
                raise ImportError("This module requires gdown. Try: `pip install gdown`")
            print("Downloading negative database...")
            FILE_ID = "1U4RXDNx_aijVDJ1oTaRKjo78Yakd3Mg4"
            gdown.download(id=FILE_ID, output=path, quiet=verbose)
        db = pd.read_csv(path)

    elif target_db == 'both':
        path = osp.join(db_dir, 'both.csv')
        if not osp.exists(path):
            try:
                import gdown
            except ImportError:
                raise ImportError("This module requires gdown. Try: `pip install gdown`")
            print("Downloading negative database...")
            FILE_ID = "189VtkbQ2bVpQlAe2UMBSzt_O4F7EyBWl"
            gdown.download(id=FILE_ID, output=path, quiet=verbose)
        db = pd.read_csv(path)

    first_time = True
    for column in activities_to_exclude:
        if column not in db.columns:
            if first_time:
                print(f"Columns in database: {', '.join(db.columns.tolist())}")
                first_time = False
            print(f"Warning: column: {column} does not exist in database. Ignoring")
        else:
            db = db[db[column] != 1].copy().reset_index()

    field = 'sequence' if sample_by == 'length' else 'smiles'
    if sample_by not in db:
        db[sample_by], n_bins = MATCHING[sample_by](db[field], n_jobs=n_jobs,
                                                    verbose=False)
        db.to_csv(path, index=False)
    else:
        if sample_by == 'mw':
            n_bins = 50
        else:
            n_bins = 10
    db[sample_by], disc = discretizer(db[sample_by].to_numpy(),
                                      n_bins=n_bins,
                                      return_discretizer=True)
    df[sample_by], _ = MATCHING[sample_by](df[sequence_field],
                                           n_jobs=n_jobs, verbose=False)
    df[sample_by] = disc.transform(df[sample_by].to_numpy().reshape(-1, 1))

    all_samples, all_wts = [], []
    for (b, g_df), (_, g_db) in zip(df.groupby(sample_by),
                                    db.groupby(sample_by)):
        pos = (g_df[label_field] == 1).sum()
        neg = (g_df[label_field] == 0).sum()
        samples = g_db.sample(n=min(int((pos - neg) * desired_ratio),
                                    len(g_db)), random_state=random_state)
        if len(samples) > 0:
            all_samples.extend(samples['smiles'].tolist())
            all_wts.extend(samples[sample_by].tolist())

    neg_df = pd.DataFrame([{'smiles': s, label_field: 0, sample_by: wt}
                           for s, wt in zip(all_samples, all_wts)])
    neg_df[sequence_field] = neg_df['smiles']
    neg_df = neg_df[[sequence_field, sample_by]]
    neg_df[label_field] = 0
    original_columns = df.columns.tolist()
    df = pd.concat([df, neg_df]).sample(frac=1).reset_index()
    return df[[*original_columns]]
