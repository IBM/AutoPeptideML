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


def get_neg_db(target_db: str, verbose: bool, return_path: bool = False) -> pd.DataFrame:
    """
    Retrieves a precompiled database of negative samples.

    If the database file is not found locally, it will attempt to download it
    using `gdown`.

    Valid values for ``target_db``:

    - ``'canonical'``: Uses the canonical negative dataset.
    - ``'non-canonical'``: Uses the non-canonical version.
    - ``'both'``: Merged version of both canonical and non-canonical datasets.

    :param target_db: The type of database to retrieve. Must be one of ``'canonical'``, ``'non-canonical'``, or ``'both'``.
    :type target_db: str
    :param verbose: If ``True``, prints information during download.
    :type verbose: bool
    :param return_path: If ``True``, also returns the path to the local database file.
    :type return_path: bool
    :raises ImportError: If `gdown` is not installed and download is required.
    :return: The negative sample database as a DataFrame, optionally with its file path.
    :rtype: pd.DataFrame or Tuple[pd.DataFrame, str]
    """
    db_dir = osp.join(osp.dirname(__file__), '..', 'data', 'dbs')
    if not osp.isdir(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    if target_db == 'canonical':
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
    if not return_path:
        return db
    else:
        return db, path


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
    """
    Augments a dataset with negative samples from a target database to achieve a desired negative/positive ratio.

    The function groups the input data and the negative database using a sampling strategy
    (e.g. molecular weight or sequence length), then balances each group individually
    by adding an appropriate number of negative samples.

    :param df: The input DataFrame containing positive samples.
    :type df: pd.DataFrame
    :param target_db: Either the name of a precompiled negative database or a custom DataFrame. Valid string values are:
                      ``'canonical'``, ``'non-canonical'``, ``'both'``.
    :type target_db: Union[str, pd.DataFrame]
    :param sequence_field: Name of the column in ``df`` containing sequences to process.
    :type sequence_field: str
    :param activities_to_exclude: A list of column names in the database to filter out active entries.
    :type activities_to_exclude: List[str]
    :param label_field: Column name to use for labels. If ``None``, defaults to ``'label'`` and assumes all input samples are positive (label=1).
    :type label_field: str, optional
    :param desired_ratio: The desired positive:negative ratio to achieve. For example, 1.0 adds one negative for each positive.
    :type desired_ratio: float
    :param verbose: Whether to print warnings and progress messages.
    :type verbose: bool
    :param sample_by: The feature used to group and sample negatives. Options:
                      - ``'mw'``: Molecular weight (computed via RDKit).
                      - ``'length'``: Sequence length.
    :type sample_by: str
    :param n_jobs: Number of parallel workers to use for feature computation.
    :type n_jobs: int
    :param random_state: Random seed for reproducibility during sampling.
    :type random_state: int
    :raises ValueError: If input parameters are invalid or required columns are missing.
    :return: A new DataFrame with the added negative samples.
    :rtype: pd.DataFrame
    """
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

    elif label_field is None:
        label_field = 'label'
        df[label_field] = 1

    if isinstance(target_db, pd.DataFrame):
        db = target_db

    else:
        db, path = get_neg_db(target_db, verbose=verbose, return_path=True)

    first_time = True
    for column in activities_to_exclude:
        if column not in db.columns:
            if first_time:
                print(f"Columns in database: {', '.join(db.columns.tolist())}")
                first_time = False
            print(f"Warning: column: {column} does not exist in database. Ignoring")
        else:
            db = db[db[column] != 1].copy().reset_index(drop=True)

    field = 'sequence' if sample_by == 'length' else 'smiles'
    if sample_by not in db:
        db[sample_by], n_bins = MATCHING[sample_by](db[field], n_jobs=n_jobs,
                                                    verbose=False)
        if not isinstance(target_db, pd.DataFrame):
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

    all_samples, all_wts, all_seqs = [], [], []
    for (b, g_df), (_, g_db) in zip(df.groupby(sample_by),
                                    db.groupby(sample_by)):
        pos = (g_df[label_field] == 1).sum()
        neg = (g_df[label_field] == 0).sum()
        samples = g_db.sample(n=min(int((pos - neg) * desired_ratio),
                                    len(g_db)), random_state=random_state)
        if len(samples) > 0:
            all_samples.extend(samples['smiles'].tolist())
            all_wts.extend(samples[sample_by].tolist())
            all_seqs.extend(samples['sequence'].tolist())

    neg_df = pd.DataFrame([{'apml-smiles': s, label_field: 0, sample_by: wt,
                            'apml-seqs': seq}
                           for s, wt, seq in zip(all_samples, all_wts,
                                                 all_seqs)])
    neg_df[sequence_field] = neg_df['apml-smiles']
    neg_df = neg_df[[sequence_field, sample_by, 'apml-seqs']]
    neg_df[label_field] = 0
    original_columns = df.columns.tolist()
    df = pd.concat([df, neg_df]).sample(frac=1).reset_index()
    return df[[*original_columns]]
