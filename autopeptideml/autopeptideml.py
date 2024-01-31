from copy import deepcopy
from multiprocessing import cpu_count
import os
from typing import Dict, List, Optional, Union 

from graph_part import stratified_k_fold
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scikitplot as skplt
import sklearn.metrics
from sklearn.model_selection import StratifiedKFold

from .data.algorithms import SYNONYMS, SUPPORTED_MODELS
from .data.metrics import METRICS, METRIC2FUNCTION, THRESHOLDED_METRICS
from .data.residues import is_canonical
from .utils.embeddings import RepresentationEngine
from .utils.training import FlexibleObjective, UniDL4BioPep_Objective, ModelSelectionObjective
from .utils.dataset_split import make_graphs_from_sequences, train_test


class AutoPeptideML:
    """
    Main class for handling the automatic development
    of bioactive peptide ML predictors.
    """
    def __init__(
        self,
        verbose: bool=True,
        threads: int=cpu_count(),
        seed: int=42
    ):
        """Initialize instance of the AutoPeptideML class

        :param verbose: Whether to output information, defaults to True
        :type verbose: bool, optional
        :param threads: Number of threads to compute parallelise
                        processes, defaults to cpu_count()
        :type threads: int, optional
        :param seed: Pseudo-random number generator seed.
                        Important for reproducibility, defaults to 42
        :type seed: int, optional
        """
        self.verbose = verbose
        self.threads = threads
        self.seed = seed

        self._welcome()
        self.db = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'data', 'peptipedia'
        )
        self.tags = self._bioactivity_tags()

    def autosearch_negatives(
        self,
        df_pos: pd.DataFrame,
        positive_tags: List[str],
        proportion: float=1.0
    ) -> pd.DataFrame:
        """Method for searching bioactive databases for peptides

        :param df_pos: DataFrame with positive peptides.
        :type df_pos: pd.DataFrame
        :param positive_tags: List of names of bioactivities
                              that may overlap with the target
                              bioactivities.
        :type positive_tags: List[str]
        :param proportion: Negative:Positive ration in
                           the new dataset. Defaults to 1.0.,
                           defaults to 1.0.
        :type proportion: float, optional
        :return: New dataset with both positive and negative
                 peptides.
        :rtype: pd.DataFrame
        """
        if self.verbose is True:
            print('\nStep 2: Autosearch for negative peptides')

        df_neg = pd.DataFrame()
        lengths = df_pos.sequence.map(len).to_numpy()
        min_length, max_length = lengths.min(), lengths.max()
        min_length = min_length - (min_length % 5)
        missing = 0

        for length in range(min_length, max_length, 5):
            subdb_path = os.path.join(self.db, f'peptipedia_{length}-{length+5}.csv')
            if os.path.isfile(subdb_path):
                subdf = pd.read_csv(subdb_path)
                subdf.drop_duplicates(subset=['sequence'], inplace=True, ignore_index=True)

                for name in positive_tags:
                    name = ' '.join(name.split('_'))
                    try:
                        subdf = subdf[subdf[name] == 0]
                    except KeyError:
                        continue

                df_len_bin = df_pos[(lengths >= length) & (lengths < length+5)]
                samples_to_draw = int(df_len_bin.shape[0] * proportion) + missing

                if samples_to_draw < len(subdf):
                    subdf = subdf.sample(
                        samples_to_draw,
                        replace=False,
                        random_state=self.seed
                    )
                    subdf.reset_index(drop=True)
                    missing = 0
                elif len(subdf) > 0:
                    subdf = subdf.sample(
                        len(subdf),
                        replace=False,
                        random_state=self.seed
                    )
                    subdf.reset_index(drop=True)
                    missing += samples_to_draw - len(subdf)
                else:
                    missing += samples_to_draw
                df_neg = pd.concat([df_neg, subdf])
        
        get_tags = lambda x: ';'.join(sorted(set(
            [column if (x[column] == 1) else df_neg.columns[0]
             for column in self.tags]
        )))

        df_neg['bioactivity'] = df_neg.apply(get_tags, axis=1)
        df_neg.drop(columns=self.tags+['is_aa_seq'], inplace=True)
        n_pos = len(df_pos)
        df_neg.reset_index(inplace=True)
        df_neg['id'] = df_neg.index.map(lambda x: n_pos + x)
        df_neg['Y'] = 0
        df = pd.concat([df_pos, df_neg]).sample(
            frac=1,
            replace=False,
            random_state=self.seed
        )
        return df.reset_index(drop=True)

    def balance_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Oversample the underrepresented class in the DataFrame.

        :param df:  DataFrame with positive and
                    negative peptides to be balanced.
        :type df: pd.DataFrame
        :return: DataFrame with balanced number of 
                 positive and negative peptides.
        :rtype: pd.DataFrame
        """
        df.Y = df.Y.map(int)
        df_pos, df_neg = deepcopy(df[df.Y == 1]), deepcopy(df[df.Y == 0])

        if len(df_pos) > len(df_neg):
            df_neg = df_neg.sample(
                len(df_pos),
                replace=True,
                random_state=self.seed,
                ignore_index=True
            )
        elif len(df_pos) < len(df_neg):
            df_pos = df_pos.sample(
                len(df_neg), 
                replace=True,
                random_state=self.seed,
                ignore_index=True
            )
        df = pd.concat([df_pos, df_neg])
        df = df.sample(len(df), random_state=self.seed)
        return df.reset_index(drop=True)

    def compute_representations(
        self,
        datasets: Dict[str, pd.DataFrame],
        re: RepresentationEngine
    ) -> dict:
        """Use a Protein Representation Model, loaded with the
        RepresentationEngine class to compute representations
        for the peptides in the `dataasets`.

        :param datasets: dictionary with the dataset partitions as DataFrames.
                        Output from the method `train_test_partition`.
        :type datasets: Dict[str, pd.DataFrame]
        :param re: class with a Protein Representation Model.
        :type re: RepresentationEngine
        :return: Dictionary with pd.DataFrame `id` column as keys and
                 the representation of the `sequence` column as values.
        :rtype: dict
        """
        if self.verbose is True:
            print('\nStep 4: PLM Peptide Featurization')
        id2rep = {}
        for df in datasets.values():
            df_repr = re.compute_representations(df.sequence, average_pooling=True)
            id2rep.update({id: repr for id, repr in zip(df.id, df_repr)})
        return id2rep

    def curate_dataset(
        self,
        dataset: Union[str, pd.DataFrame],
        outputdir: str = None
    ) -> pd.DataFrame:
        """Load a DataFrame or use one already loaded and then remove
        all entries with non-canonical residues or repeated sequences.

        :param dataset: Dataset or path to dataset.
        :type dataset: Union[str, pd.DataFrame]
        :param outputdir: Path were to save the curated dataset, defaults to None
        :type outputdir: str, optional
        :return: Curated dataset.
        :rtype: pd.DataFrame
        """
        if self.verbose is True:
            print('\nStep 1: Dataset curation')

        if isinstance(dataset, pd.DataFrame):
           df = dataset

        elif dataset.endswith('.fasta'):
            if outputdir is None:
                pass
            elif not os.path.isdir(outputdir):
                os.mkdir(outputdir)
            df = self._fasta2csv(dataset, outputdir)
        else:
            df = pd.read_csv(dataset)

        if not ('y' in df.columns or 'Y' in df.columns):
            df['Y'] = 1

        df.drop_duplicates('sequence', inplace=True, ignore_index=True)
        df = df[df.sequence.map(is_canonical)].reset_index(drop=True)
        df = df[~pd.isna(df.sequence)]
        return df

    def evaluate_model(
        self,
        best_model: list,
        test_df: pd.DataFrame,
        id2rep: dict,
        outputdir: str
    ) -> pd.DataFrame:
        """Evaluate an ensemble model.

        :param best_model:  List of models with a `predict_proba` method.
        :type best_model: list
        :param test_df: Evaluation dataset with `id`, `sequence`
                        and `Y` columns.
        :type test_df: pd.DataFrame
        :param id2rep: Dictionary with keys being the `id` and the values
                       the peptide representations.
        :type id2rep: dict
        :param outputdir: Path were to save the evaluation data.
        :type outputdir: str
        :return: Dataset with the evaluation metrics.
        :rtype: pd.DataFrame
        """
        if self.verbose is True:
            print('\nStep 6: Model evaluation')
        raw_data_path = os.path.join(outputdir, 'evaluation_data')
        figures_path = os.path.join(outputdir, 'figures')
        paths = [outputdir, raw_data_path, figures_path]
        for path in paths:
            if not os.path.isdir(path):
                os.mkdir(path)
        
        embds = np.stack([id2rep[id] for id in test_df.id])
        truths = np.array(test_df.Y.tolist())

        scores = []
        preds_proba = np.zeros(len(test_df))
        for idx, clf in enumerate(best_model['estimators']):
            preds_proba = preds_proba + clf.predict_proba(embds)[:, 1] * (1/len(best_model['estimators']))
        
        preds = preds_proba > 0.5

        for metric in METRICS:
            if metric in THRESHOLDED_METRICS:
                score = METRIC2FUNCTION[metric](truths, preds)
            else:
                score = METRIC2FUNCTION[metric](truths, preds_proba)
            scores.append({'metric': metric, 'score': score})

        confusion_matrix = sklearn.metrics.confusion_matrix(truths, preds)
        scores.append({'metric': 'TP', 'score': confusion_matrix[0, 0]})
        scores.append({'metric': 'TN', 'score': confusion_matrix[1, 1]})
        scores.append({'metric': 'FP', 'score': confusion_matrix[0, 1]})
        scores.append({'metric': 'FN', 'score': confusion_matrix[1, 0]})
      
        self._make_figures(figures_path, truths, preds_proba)
        df = pd.DataFrame(scores)
        df.to_csv(os.path.join(raw_data_path, 'test_scores.csv'), index=False, float_format="{:.4f}".format)
        self._summary(outputdir)
        return df

    def hpo_train(
        self,
        config: dict,
        train_df: pd.DataFrame,
        id2rep: dict,
        folds: list,
        outputdir: str,
        n_jobs: int = 1
    ) -> list:
        """Hyperparameter Optimisation and training.

        :param config: dictionary with hyperparameter search space.
        :type config: dict
        :param train_df: Training dataset with `id` column and `Y` column
                         with the bioactivity target.
        :type train_df: pd.DataFrame
        :param id2rep: Dictionary with pd.DataFrame `id` column as keys and
                       the representation of the `sequence` column as values.
        :type id2rep: dict
        :param folds: List with the training/validation folds
        :type folds: list
        :param outputdir: Path to the directory where information should be
                          saved.
        :type outputdir: str
        :param n_jobs: Number of threads to parallelise the training,
                       defaults to 1.
        :type n_jobs: int, optional
        :return: List with the models that comprise the final ensemble.
        :rtype: list
        """
        if self.verbose is True:
            print('\nStep 5: Hyperparameter Optimisation and Model Training')
        np.random.seed(self.seed)
        best_configs_path = os.path.join(outputdir, 'best_configs')
        best_ensemble_path = os.path.join(outputdir, 'ensemble')
        evaluation_path = os.path.join(outputdir, 'evaluation_data')
        paths = [outputdir, best_configs_path, best_ensemble_path, evaluation_path]
        for path in paths:
            if not os.path.exists(path):
                os.mkdir(path)

        objectives = {}

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if 'ensemble' in config:
            for study in config['ensemble']:
                optuna_study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.RandomSampler(seed=self.seed)
                )
                if study['model'] in SYNONYMS['unidl4biopep']:
                    objective = UniDL4BioPep_Objective(
                        study,
                        train_df,
                        folds,
                        id2rep,
                        self.threads,
                        outputdir
                    )
                else:
                    objective = FlexibleObjective(
                        study,
                        train_df,
                        folds,
                        id2rep,
                        self.threads,
                        outputdir
                    )
                optuna_study.optimize(
                    objective,
                    n_trials=study['trials'],
                    callbacks=[objective.callback],
                    n_jobs=n_jobs,
                    show_progress_bar=self.verbose
                )
                objectives[study['model']] = objective
        elif 'model_selection':
            optuna_study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.RandomSampler(seed=self.seed)
            )
            objective = ModelSelectionObjective(
                config['model_selection'],
                train_df,
                folds,
                id2rep,
                self.threads,
                outputdir
            )
            optuna_study.optimize(
                objective,
                n_trials=config['trials'],
                callbacks=[objective.callback],
                n_jobs=n_jobs,
                show_progress_bar=self.verbose
            )
            objectives['model_selection'] = objective


        output = {}

        for name, objective in objectives.items():
            output[name] = {}
            for key, value in objective.model.items():
                if isinstance(value, np.ndarray):
                    output[name][key] = value.tolist() 
                    continue
                for model_class in SUPPORTED_MODELS.values():
                    if not isinstance(value, model_class):
                        continue
                    output[name][key] = value.get_params()

        df_output = []
        for i, fold in enumerate(folds):
            for name in output.keys():
                entry = {'model': name, 'fold': i}
                for metric in METRICS:
                    entry[metric] = output[name][metric][i]
                df_output.append(entry)
        
        df_output = pd.DataFrame(df_output)
        df_output.to_csv(os.path.join(evaluation_path, 'cross-validation.csv'), index=False, float_format='{:.4f}'.format)

        output = {}
        output['estimators'] = []

        for name in objectives.keys():
            if name == 'unidl4biopep':
                output['estimators'].append(objectives[name].best_model)
            else:
                output['estimators'].extend(objectives[name].best_model['estimators'])
        return output

    def train_test_partition(
        self,
        df: pd.DataFrame,
        threshold: float=0.3,
        test_size: float=0.2,
        alignment: str='mmseqs+prefiler',
        outputdir: str='./splits'
    ) -> Dict[str, pd.DataFrame]:
        """Novel homology partitioning algorithm for generating
        independent hold-out evaluation sets.

        :param df: Dataset to partition with the following columns
                   `id`, `sequence`, and `labels`.
        :type df: pd.DataFrame
        :param threshold: Maximum sequence identity allowed between sequences
                          in training and evaluation sets, defaults to 0.3
        :type threshold: float, optional
        :param test_size: Proportion of samples in evaluation set, defaults to 0.2
        :type test_size: float, optional
        :param alignment: Alignment algorithm to use. Options available: 
                          `mmseqs` (local Smith-Waterman alignment), `mmseqs+prefilter`
                          (local fast alignment Smith-Waterman + k-mer prefiltering),
                          and `needle` (global Needleman-Wunch alignment),
                          defaults to 'mmseqs+prefiler'
        :type alignment: str, optional
        :param outputdir: Path were information should be stored, defaults to './splits'
        :type outputdir: str, optional
        :return: Dictionary with keys `train` and `test` and values the corresponding
                 DataFrames.
        :rtype: Dict[str, pd.DataFrame]
        """
        if self.verbose == True:
            print('\nStep 3a: Dataset Partitioning (Train/Test)')
        np.random.seed(self.seed)
        os.makedirs(outputdir, exist_ok=True)

        df = df.sample(len(df), random_state=self.seed).reset_index(drop=True)

        g, seqs, labels = make_graphs_from_sequences(
            df,
            verbose=2 if self.verbose else 0,
            alignment=alignment,
            outputdir=os.path.join(outputdir, 'tmp'),
            denominator='longest',
            threshold=threshold,
            threads=self.threads
        )

        train, test = train_test(
            g=g,
            ids=df.id,
            sequences=seqs,
            labels=labels,
            test_size=test_size,
            threshold=threshold,
            verbose=2 if self.verbose else 0
        )

        train.to_csv(os.path.join(outputdir, 'train.csv'), index=False)
        test.to_csv(os.path.join(outputdir, 'test.csv'), index=False)

        return {'train': train, 'test': test}

    def train_val_partition(
        self,
        df: pd.DataFrame,
        method: str = 'random',
        threshold: float = 0.5,
        alignment: str = 'mmseqs+prefilter',
        n_folds: int = 10,
        outputdir: str = './folds'
    ) -> list:
        """Method for generating `n` training/validation folds for
        cross-validation.

        :param df: Training dataset with `id`, `sequence`, and `Y` columns.
        :type df: pd.DataFrame
        :param method: Method for generating the folds. Options available:
                       `random` through `sklearn.model_selection.StratifiedKFold` or
                       `graph-part` through `graphpart.stratified_k_fold`, defaults to
                       `random`.
        :type method: str
        :param threshold: If mode is `graph-part`, maximum sequence identity allowed
                          between sequences
                          in training and evaluation sets, defaults to 0.5
        :type threshold: float, optional
        :param alignment: If mode is `graph-part`,
                          alignment algorithm to use. Options available: 
                          `mmseqs` (local Smith-Waterman alignment), `mmseqs+prefilter`
                          (local fast alignment Smith-Waterman + k-mer prefiltering),
                          and `needle` (global Needleman-Wunch alignment),
                          defaults to 'mmseqs+prefiler', defaults to 'mmseqs+prefilter'
        :type alignment: str, optional
        :param n_folds: Number of training/validation folds to generate, defaults to 10
        :type n_folds: int, optional
        :param outputdir: Path where data should be saved, defaults to './folds'
        :type outputdir: str, optional
        :return: List of training/validation folds
        :rtype: list
        """
        if self.verbose:
            print('\nStep 3b: Dataset Partitioning (Train/Val)')
        np.random.seed(self.seed)
        os.makedirs(outputdir, exist_ok=True)
        prefilter = False
        df = df.sample(len(df), random_state=self.seed).reset_index(drop=True)
        if method == 'random':
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
            x, y = df.index.to_numpy(), df.Y.to_numpy()
            fold_ids = [fold[1] for fold in kf.split(x, y)]
        elif method == 'graph-part':
            if 'mmseqs' in alignment:
                alignment = alignment.replace('mmseqs', 'mmseqs2')
            if 'prefilter' in alignment:
                alignment = alignment.strip('+prefilter')
                prefilter = True
            fold_ids = stratified_k_fold(
                df.sequence.to_numpy(),
                labels=df.Y.to_numpy(),
                alignment_mode=alignment,
                threads=self.threads,
                chunks=self.threads,
                threshold=threshold,
                partitions=n_folds,
                remove_same=False,
                denominator='longest',
                prefilter=prefilter
            )

        folds = []
        for i, fold in enumerate(fold_ids):
            train_df = df.iloc[~df.index.isin(fold)]
            val_df = df.iloc[fold]
            train_df.to_csv(
                os.path.join(outputdir, f'train_{i}.csv'),
                index=False
            )
            val_df.to_csv(
                os.path.join(outputdir, f'val_{i}.csv'),
                index=False
            )
            folds.append({'train': train_df, 'val': val_df})
        return folds

    def predict(
        self,
        df: pd.DataFrame,
        re: RepresentationEngine,
        ensemble_path: str,
        outputdir: str
    ) -> pd.DataFrame:
        if self.verbose is True:
            print('Step 7: Prediction')
        if not os.path.isdir(outputdir):
            os.mkdir(outputdir)
        output_path = os.path.join(outputdir, 'predictions.csv')
        df_repr = re.compute_representations(df.sequence, average_pooling=True)
        df_repr = np.stack(df_repr)

        predictions = []
        for model in os.listdir(ensemble_path):
            model_path = os.path.join(ensemble_path, model)
            clf = joblib.load(model_path)
            predictions.append(clf.predict(df_repr))

        predictions = np.stack(predictions)
        predictions = np.mean(predictions, axis=0)
        df['prediction'] = predictions
        df.sort_values(by='prediction', inplace=True, ascending=False, ignore_index=True)
        
        df.to_csv(output_path, index=False)
        return df

    @classmethod
    def _bioactivity_tags(self) -> list:
        tags = []
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'data', 'bioactivities.txt'
        )
        with open(path) as file:
            for line in file:
                tags.append(line.strip('\n'))
        return tags

    def _fasta2csv(self, dataset: str, outputdir: str) -> pd.DataFrame:
        data = []
        with open(dataset) as file:
            for idx, line in enumerate(file):
                line = line.strip('\n')
                if idx % 2 == 0:
                    try:
                        data.append({'Y': int(line[1:])})
                    except ValueError:
                        data.append({'Y': 1, 'id': line[1:]})
                elif idx % 2 == 1:
                    if '/' in line:
                        data.pop()
                    data[-1]['sequence'] = line
        
        output_path = os.path.join(
            outputdir,
            f"{dataset.split('/')[-1].split('.')[0]}.csv"
        )
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, sep=',')
        return df

    def _make_figures(self, figures_path: str, truths, preds_proba):
        preds = preds_proba > 0.5
        new_preds_proba = np.zeros((len(preds_proba), 2))
        new_preds_proba[:, 0] = 1 - preds_proba
        new_preds_proba[:, 1] = preds_proba
        preds_proba = new_preds_proba
        skplt.metrics.plot_confusion_matrix(truths, preds, normalize=False, 
                                            title='Confusion Matrix')
        plt.savefig(os.path.join(figures_path, 'confusion_matrix.png'))
        plt.close()
        skplt.metrics.plot_roc(truths, preds_proba, title='ROC Curve', plot_micro=False, plot_macro=False, classes_to_plot=[1])
        plt.savefig(os.path.join(figures_path, 'roc_curve.png'))
        plt.close()
        skplt.metrics.plot_precision_recall(truths, preds_proba, title='Precision-Recall Curve', plot_micro=False, classes_to_plot=[1])
        plt.savefig(os.path.join(figures_path, 'precision_recall_curve.png'))
        plt.close()
        skplt.metrics.plot_calibration_curve(truths, [preds_proba], title='Calibration Curve')
        plt.savefig(os.path.join(figures_path, 'calibration_curve.png'))
        plt.close()

    def _welcome(self) -> None:
        if self.verbose is True:
            message = '| Welcome to AutoPeptideML |'
            print('-' * len(message))
            print(message)
            print('-' * len(message))

    def _summary(self, outputdir: str) -> None:
        metrics = {
            "- **Accuracy:**": "accuracy",
            "- **Sensitivity or recall:**": "recall",
            "- **Specificity or precision:**": "precision",
            "- **F1:**": "f1",
            "- **Matthew's correlation coefficient:**": "matthews_corrcoef"
        }
        df = pd.read_csv(os.path.join(outputdir, 'evaluation_data', 'test_scores.csv'))
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'data', 'readme_ex.md'
        )
        new_lines = []
        with open(path) as file:
            for line in file:
                if line.strip() in metrics.keys():
                    new_value = df.loc[df['metric'] == metrics[line.strip('\n')], 'score'].tolist()[0]
                if '    - *Value:*' in line:
                    new_lines.append(line.strip('\n') + ' `' + str(round(new_value, 3)) + '`\n')
                else:
                    new_lines.append(line)
        
        new_readme = ''.join(new_lines)
        summary_path = os.path.join(outputdir, 'summary')
        with open(f"{summary_path}.md", 'w') as file:
            file.write(new_readme)
        
        os.system(f"mdpdf -o {summary_path}.pdf {summary_path}.md")
        os.remove(f"{summary_path}.md")
        os.remove("mdpdf.log")
