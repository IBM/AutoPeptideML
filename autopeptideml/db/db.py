import os
import os.path as osp

from typing import *

import pandas as pd
import numpy as np

from ..pipeline import Pipeline
from ..pipeline.smiles import is_smiles

from tqdm import tqdm


DBS = {
    'apml-peptipedia2.csv': 'https://drive.usercontent.google.com/uc?id=189VtkbQ2bVpQlAe2UMBSzt_O4F7EyBWl',
    'Gonzalez_2023_NC_PeptideDB.csv':  'https://drive.usercontent.google.com/uc?id=1U4RXDNx_aijVDJ1oTaRKjo78Yakd3Mg4',
    'apml-pep2+Gonzalez.csv': ''
}


class Database:
    """
    Class that handles dataset operations within AutoPeptideML.
    """
    df: pd.DataFrame
    # Pipeline can be a single pipeline or a dictionary of field - Pipeline
    pipe: Union[Pipeline, Dict[str, Pipeline]]
    # Feat_fields can be a single field or a list of fields (e.g, ['seq', 'smiles'])
    feat_fields: Union[str, List[str]]
    label_field: Optional[str]

    def __init__(
        self,
        path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        feat_fields: Union[str, List[str]] = None,
        pipe: Optional[Union[Pipeline, Dict[str, Pipeline]]] = None,
        label_field: Optional[str] = None,
        verbose: bool = False,
        seed: int = 1
    ):
        """Initialises a Database instance.

        :type path: Optional[str]
        :param path: Path to the CSV file containing the dataset. If provided, the dataset will be loaded from this path.

        :type df: Optional[pd.DataFrame]
        :param df: The dataset represented as a pandas DataFrame. If `path` is provided, this will be ignored.

        :type pipe: Union[Pipeline, Dict[str, Pipeline]]
        :param pipe: A preprocessing pipeline or a dictionary of feature fields mapped to their respective pipelines. 
                    If not provided, no preprocessing is applied.

        :type feat_fields: Union[str, List[str]]
        :param feat_fields: A single feature field or a list of feature fields (e.g., `['seq', 'smiles']`) 
                            used for processing and model input. This parameter is required.

        :type label_field: Optional[str]
        :param label_field: The name of the column representing labels in the dataset. If `None`, no label column is specified.

        :type verbose: bool
        :param verbose: Enables verbose output if set to `True`. Logs detailed preprocessing steps. Default is `False`.

        """
        if path is not None:
            if not osp.exists(path) and osp.basename(path) in DBS:
                url = DBS[osp.basename(path)]
                if not osp.isdir(osp.dirname(path)):
                    os.makedirs(osp.dirname(path), exist_ok=True)

                print("Downloading negative database...")
                import gdown
                gdown.download(url, path, quiet=True)

            self.df = pd.read_csv(path)
        else:
            self.df = df
        if feat_fields is None:
            raise ValueError(f'`feat_fields` cannot be left empty')
        if isinstance(feat_fields, str):
            feat_fields = [feat_fields]
        if (not isinstance(pipe, dict) and pipe is not None):
            self.pipe = {field: pipe for field in feat_fields}
        else:
            self.pipe = pipe
        self.seed = seed
        self.label_field = label_field
        self.feat_fields = feat_fields
        self.verbose = verbose
        self._preprocess(verbose)

    def draw_samples(
        self,
        target_db: "Database",
        columns_to_exclude: Optional[Union[List[str], str]] = None
    ) -> pd.DataFrame:
        """
        Draws samples from the current database to match the distribution of the target database. 
        Excludes specified columns if provided.

        :type target_db: Database
          :param target_db: The target `Database` whose distribution is used to sample data.

        :type columns_to_exclude: Optional[Union[List[str], str]]
          :param columns_to_exclude: A single column or list of columns to exclude from sampling. If `None`, no columns are excluded.

        :rtype: pd.DataFrame
          :return: A DataFrame containing the sampled data matching the target database distribution.
        """
        if columns_to_exclude is not None:
            self._filter(columns_to_exclude)

        target_hist = target_db._hist()
        hist = self._hist()

        entries = {field: [] for field in self.feat_fields}
        left_out = 0
        for idx, h in enumerate(target_hist):
            if idx > len(hist):
                break
            tmp_df = self.df.iloc[hist[idx]]
            tgt_df = target_db.df.iloc[h]

            if len(tmp_df) < len(tgt_df):
                left_out += len(tgt_df) - len(tmp_df)
            elif len(tmp_df) > len(tgt_df) + np.abs(left_out):
                if left_out < 0:
                    smp = len(tgt_df)
                else:
                    smp = len(tgt_df) + left_out
                tmp_df = tmp_df.sample(smp, replace=False, random_state=self.seed)
                left_out = 0
            else:
                smp = len(tmp_df) - len(tgt_df)
                tmp_df = tmp_df.sample(smp, replace=False, random_state=self.seed)
            for field in self.feat_fields:
                entries[field].extend(tmp_df[field].tolist())

        entries_df = pd.DataFrame(entries)
        for field in self.feat_fields:
            entries_df.drop_duplicates(field, inplace=True)
        return entries_df

    def add_negatives(
        self, other: "Database",
        columns_to_exclude: Optional[Union[List[str], str]] = None
    ):
        """
        Adds negative samples to the current database using another database. 
        The label for negative samples is set to `0`.

        :type other: Database
          :param other: The source `Database` from which negative samples are drawn.

        :type columns_to_exclude: Optional[Union[List[str], str]]
          :param columns_to_exclude: A single column or list of columns to exclude during sampling. If `None`, no columns are excluded.

        :rtype: None
          :return: Updates the current database with the added negative samples.
        """
        other.df = other.draw_samples(self, columns_to_exclude)
        if self.label_field is None:
            self.label_field = "Y"
            self.df[self.label_field] = 1

        other.df[self.label_field] = 0
        if other.feat_fields[0] != self.feat_fields[0]:
            other.df[self.feat_fields[0]] = other.df[other.feat_fields[0]]
        self.df = pd.concat([self.df, other.df])
        self.df = self.df[[self.label_field, *self.feat_fields]]

    def _check_fields(self):
        """
        Validates that all feature fields exist in the dataset.

        :rtype: None
          :return: Raises a `KeyError` if any feature field is missing from the dataset.
        """
        for field in self.feat_fields:
            if field not in self.df.columns:
                raise KeyError(
                    f"Field: {field} is not in df",
                    f"df columns are: {', '.join(self.df.columns.tolist())}"
                )

    def _get_mw(self):
        """
        Computes the molecular weight (MW) for each entry in the dataset using RDKit.

        :rtype: None
          :return: Adds a `tmp_mw` column to the dataset with computed molecular weights.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
        except ImportError:
            raise ImportError("Rdkit is required for this function",
                              "Please install: `pip install rdkit`")
        item = self.df.iloc[0, :]
        for field in self.feat_fields:
            if is_smiles(item[field]):
                self.df['tmp_mw'] = self.df[field].map(
                    lambda x: Descriptors.ExactMolWt(
                        Chem.MolFromSmiles(x)
                    )
                )
            else:
                self.df['tmp_mw'] = self.df[field].map(
                    lambda x: Descriptors.ExactMolWt(
                        Chem.MolFromFASTA(x)
                    )
                )

    def _preprocess(self, verbose):
        """
        Applies preprocessing steps to the dataset, including field validation and pipeline execution.

        :type verbose: bool
          :param verbose: Enables verbose output if set to `True`.

        :rtype: None
          :return: Updates the dataset with preprocessed feature fields.
        """
        self._check_fields()
        if verbose:
            print("Preprocessing database")
        if self.pipe is not None:
            for field in self.feat_fields:
                self.df[field] = self.pipe[field](self.df[field], verbose=verbose)
        self._get_mw()

    def _filter(self, columns: Union[List[str], str]):
        """
        Filters out rows where specified columns contain the value `1`.

        :type columns: Union[List[str], str]
          :param columns: A single column or list of columns to filter.

        :rtype: None
          :return: Updates the dataset after filtering.
        """
        if isinstance(columns, str):
            columns = [columns]
        for column in columns:
            self.df = self.df[self.df[column] != 1].copy().reset_index(drop=True)

    def _hist(self) -> List[np.ndarray]:
        """
        Creates histograms based on molecular weight ranges for the dataset.

        :rtype: List[np.ndarray]
          :return: A list of boolean arrays indicating the molecular weight bins.
        """
        av_mw_aa = 110
        step = 5 * av_mw_aa
        max_mw = int(self.df['tmp_mw'].max())
        out = []
        if self.verbose:
            pbar = tqdm(range(0, max_mw, step), desc='Computing MW')
        else:
            pbar = range(0, max_mw, step)
        for mw in pbar:
            cond = ((self.df.tmp_mw > mw) & (self.df.tmp_mw <= mw + step)).to_numpy()
            cond = cond.astype(np.bool_)
            out.append(cond)
        return out

    def __len__(self) -> int:
        """
        Returns the number of rows in the dataset.

        :rtype: int
          :return: The number of rows in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> pd.Series:
        """
        Retrieves a row from the dataset by index, returning only the feature fields and the label field, if specified.

        :type idx: int
          :param idx: The index of the row to retrieve.

        :rtype: pd.Series
          :return: A series containing the feature fields and the label field if specified for the specified row.
        """
        item = self.df.iloc[idx]
        if self.label_field is None:
            return item[self.feat_fields]
        else:
            return item[self.feat_fields + self.label_field]

    def __str__(self) -> str:
        return str(self.df.head())
