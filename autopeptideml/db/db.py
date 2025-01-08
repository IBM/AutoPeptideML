from typing import *

import pandas as pd
import numpy as np

from ..pipeline import Pipeline
from ..pipeline.smiles import is_smiles

from tqdm import tqdm


class Database:
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
        verbose: bool = False
    ):
        if path is not None:
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
        self.label_field = label_field
        self.feat_fields = feat_fields
        self._preprocess(verbose)

    def draw_samples(
        self,
        target_db: "Database",
        columns_to_exclude: Optional[Union[List[str], str]] = None
    ) -> pd.DataFrame:
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
                tmp_df = tmp_df.sample(smp, replace=False)
                left_out = 0
            else:
                smp = (left_out + len(tgt_df)) - len(tmp_df)
                tmp_df = tmp_df.sample(smp, replace=False)
            for field in self.feat_fields:
                entries[field].extend(tmp_df[field].tolist())

        entries_df = pd.DataFrame(entries)
        for field in self.feat_fields:
            entries_df.drop_duplicates(field, inplace=True)
        return entries_df

    def add_negatives(self, other: "Database"):
        other[self.label_field] = 0
        self.df = pd.concat([self.df, other])

    def _check_fields(self):
        for field in self.feat_fields:
            if field not in self.df.columns:
                raise KeyError(
                    f"Field: {field} is not in df",
                    f"df columns are: {', '.join(self.df.columns.tolist())}"
                )

    def _get_mw(self):
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
        self._check_fields()
        if verbose:
            print("Preprocessing database")
        if self.pipe is not None:
            for field in self.feat_fields:
                self.df[field] = self.pipe[field](self.df[field], verbose=verbose)
        self._get_mw()

    def _filter(self, columns: Union[List[str], str]):
        if isinstance(columns, str):
            columns = [columns]
        for column in columns:
            self.df = self.df[self.df[column] != 1].copy().reset_index(drop=True)

    def _hist(self) -> List[np.ndarray]:
        av_mw_aa = 110
        step = 5 * av_mw_aa
        max_mw = int(self.df['tmp_mw'].max())
        out = []
        for mw in tqdm(range(0, max_mw, step)):
            cond = ((self.df.tmp_mw > mw) & (self.df.tmp_mw <= mw + step)).to_numpy()
            cond = cond.astype(np.bool_)
            out.append(cond)
        return out

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> pd.Series:
        item = self.df.iloc[idx]
        return item[self.feat_fields]
