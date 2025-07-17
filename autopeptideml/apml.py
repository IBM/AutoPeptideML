import os
import os.path as osp
import shutil
import subprocess as sp
import time
import yaml
import pickle

from datetime import datetime
from multiprocessing import cpu_count
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from hestia import HestiaGenerator, SimArguments
from hestia import __version__ as hestia_version

from .db import add_negatives_from_db
from .pipeline import get_pipeline, Pipeline
from .train import OptunaTrainer
from .train.architectures import ALL_MODELS
from .train.metrics import evaluate
from .reps import RepEngineBase, PLMs, CLMs, FPs


__version__ = '2.0.2'


class AutoPeptideML:
    """
    Main class for the AutoPeptideML pipeline that handles data preprocessing,
    negative sampling, model building, representation generation, and evaluation.

    :param data: Input data as a DataFrame or a list of peptide sequences.
    :type data: Union[pandas.DataFrame, List[str]]
    :param outputdir: Path to the output directory.
    :type outputdir: str
    :param sequence_field: Name of the column containing sequences.
    :type sequence_field: str, optional
    :param label_field: Name of the column containing labels.
    :type label_field: str, optional
    :rtype: AutoPeptideML
    """
    df: pd.DataFrame
    metadata: dict = {}
    parts = None
    x: dict = {}
    execution: dict = {}
    hpo_run: int = 1

    def __init__(
        self,
        data: Union[pd.DataFrame, List[str]],
        outputdir: str,
        sequence_field: str = None,
        label_field: str = None
    ):
        outputdir = osp.join(outputdir, self._get_current_timestamp())
        self.outputdir = outputdir.replace(' ', '_')
        self.meta_dir = osp.join(self.outputdir, 'metadata')
        os.makedirs(self.outputdir, exist_ok=False)
        os.makedirs(self.meta_dir, exist_ok=True)

        if isinstance(data, list):
            if sequence_field is None:
                sequence_field = 'peptide'
            if label_field is None:
                label_field = 'label'

            self.df = pd.DataFrame({sequence_field: data,
                                    label_field: [1 for _ in data]})
        else:
            self.df = data.copy()

        self.sequence_field = sequence_field
        self.label_field = label_field
        self.df.to_csv(osp.join(self.meta_dir, 'start-data.tsv'),
                       index=False,
                       sep='\t')
        self.df.drop_duplicates(subset=self.sequence_field, inplace=True)
        self.metadata['start-time'] = self._get_current_timestamp()
        self.metadata['outputdir'] = self.outputdir
        self.metadata['autopeptideml-version'] = __version__
        self.metadata['size'] = len(self.df)
        self.metadata['sequence-field'] = self.sequence_field
        self.metadata['label-field'] = self.label_field
        self.metadata['removed-entries'] = len(data) - len(self.df)
        self.metadata['status'] = 'started'
        self.p_it = 1
        self.save_metadata()

    def _get_current_timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save_metadata(self):
        self.metadata['last-update'] = self._get_current_timestamp()
        path = osp.join(self.meta_dir, 'metadata.yml')
        yaml.safe_dump(self.metadata, open(path, 'w'))

    def preprocess_data(
        self,
        pipeline: Union[str, Pipeline],
        n_jobs: int = cpu_count(),
        verbose: bool = True,
    ):
        """
        Apply a preprocessing pipeline to the input sequences.

        :param pipeline: Pipeline object or string identifier of a predefined pipeline.
        :type pipeline: Union[str, Pipeline]
        :param n_jobs: Number of parallel jobs to use.
        :type n_jobs: int
        :param verbose: If True, prints detailed progress.
        :type verbose: bool
        :rtype: None
    """
        self.metadata['status'] = f'preprocessing-{self.p_it}'
        if isinstance(pipeline, str):
            pipeline = get_pipeline(pipeline)

        self.metadata[f'pipeline-{self.p_it}'] = pipeline.to_dict()
        self.save_metadata()
        start = time.time()
        if self.p_it == 1:
            self.df[self.sequence_field] = pipeline(
                self.df[self.sequence_field], n_jobs=n_jobs, verbose=verbose
            )
        else:
            self.df[f'{self.sequence_field}-{self.p_it}'] = pipeline(
                self.df[self.sequence_field], n_jobs=n_jobs, verbose=verbose
            )
        end = time.time()
        self.df.to_csv(osp.join(self.outputdir, 'data.tsv'), index=False,
                       sep='\t')
        self.metadata['status'] = 'preprocessed'
        self.metadata[f'preprocessing-metadata-{self.p_it}'] = {
            'n-jobs': n_jobs,
            'execution-time': end - start
        }
        self.save_metadata()

    def sample_negatives(
        self,
        target_db: Union[str, pd.DataFrame],
        activities_to_exclude: List[str] = [],
        desired_ratio: float = 1.0,
        verbose: bool = True,
        sample_by: str = 'mw',
        n_jobs: int = cpu_count(),
        random_state: int = 1
    ):
        """
        Sample negative examples from a database to balance the dataset.

        :param target_db: Path to a database or a DataFrame to sample from.
        :type target_db: Union[str, pandas.DataFrame]
        :param activities_to_exclude: List of activities to exclude during sampling.
        :type activities_to_exclude: List[str]
        :param desired_ratio: Desired negative-to-positive ratio.
        :type desired_ratio: float
        :param verbose: If True, prints detailed progress.
        :type verbose: bool
        :param sample_by: Strategy to sample negatives ('mw' for molecular weight).
        :type sample_by: str
        :param n_jobs: Number of parallel jobs to use.
        :type n_jobs: int
        :param random_state: Random seed for reproducibility.
        :type random_state: int
        :rtype: None
        """
        if isinstance(target_db, pd.DataFrame):
            path = osp.join(self.outputdir, 'neg-db.tsv')
            target_db.to_csv(path, sep='\t', index=False)
        else:
            path = target_db

        self.metadata['status'] = 'sampling-negatives'
        self.metadata['negative-sampling-metadata'] = {
            'n-jobs': n_jobs,
            'random-state': random_state,
            'desired-ratio': desired_ratio,
            'activities-to-exclude': activities_to_exclude,
            'sample-by': sample_by,
            'target-db': path
        }
        self.save_metadata()
        start = time.time()
        self.df = add_negatives_from_db(
            df=self.df,
            label_field=self.label_field,
            sequence_field=self.sequence_field,
            target_db=target_db,
            activities_to_exclude=activities_to_exclude,
            desired_ratio=desired_ratio,
            verbose=verbose,
            sample_by=sample_by,
            n_jobs=n_jobs,
            random_state=random_state
        )
        end = time.time()
        neg = (self.df[self.label_field] == 0).sum()
        pos = (self.df[self.label_field] == 1).sum()
        self.metadata['negative-sampling-metadata'].update({
            'real-ratio': float(neg / pos),
            'execution-time': end - start,
        })
        self.metadata['original-size'] = self.metadata['size']
        self.metadata['size'] = len(self.df)
        self.metadata['status'] = 'negatives-sampled'
        path = osp.join(self.outputdir, 'data.tsv')
        self.df.to_csv(path, sep='\t', index=False)
        self.p_it += 1
        self.save_metadata()

    def build_models(
        self,
        task: str = 'class',
        ensemble: bool = False,
        reps: Union[str, List[str], Dict[str, RepEngineBase]] = ['ecfp-16'],
        models: Union[str, List[str]] = ALL_MODELS,
        split_strategy: str = 'min',
        hestia_generator: HestiaGenerator = None,
        model_configs: Dict[str, dict] = {},
        partitions: Dict[str, np.ndarray] = None,
        n_folds_cv: int = 5,
        verbose: bool = True,
        n_trials: int = 100,
        device: str = 'cpu',
        random_state: int = 1,
        n_jobs: int = cpu_count()
    ):
        """
        Build, train, and evaluate machine learning models using specified representations.

        :param task: Type of task ('class' or 'reg').
        :type task: str
        :param ensemble: If True, use an ensemble model.
        :type ensemble: bool
        :param reps: Representations to use for features.
        :type reps: Union[str, List[str], Dict[str, RepEngineBase]]
        :param models: Models to include in the hyperparameter optimization.
        :type models: Union[str, List[str]]
        :param split_strategy: Strategy for data partitioning ('random', 'min', etc.).
        :type split_strategy: str
        :param hestia_generator: HestiaGenerator instance for advanced partitioning.
        :type hestia_generator: HestiaGenerator
        :param model_configs: Custom model configuration dictionary.
        :type model_configs: Dict[str, dict]
        :param partitions: Optional pre-defined train/test partitions.
        :type partitions: Dict[str, numpy.ndarray]
        :param n_folds_cv: Number of folds for cross-validation.
        :type n_folds_cv: int
        :param verbose: If True, prints detailed progress.
        :type verbose: bool
        :param n_trials: Number of optimization trials.
        :type n_trials: int
        :param device: Device to use ('cpu', 'cuda', or 'mps').
        :type device: str
        :param random_state: Random seed.
        :type random_state: int
        :param n_jobs: Number of parallel jobs to use.
        :type n_jobs: int
        :rtype: None
        """
        if task not in ['class', 'reg']:
            raise ValueError(f"Task: {task} is not valid.",
                             "Choose one: `class, reg`")

        if split_strategy == 'good':
            raise NotImplementedError(
                "Split strategy: `good` is currently not implemented.",
                "Please try: `min`",
                "`good` strategy will be implemented in future releases."
            )
        if os.isdir(osp.join(self.outputdir, 'ensemble')):
            shutil.rmtree(osp.join(self.outputdir, 'ensemble'))
        self._partitioning(
            split_strategy=split_strategy,
            hestia_generator=hestia_generator,
            verbose=verbose,
            n_jobs=n_jobs,
            random_state=random_state,
            partitions=partitions
        )
        self._representing(
            reps=reps,
            device=device,
            n_jobs=n_jobs,
            verbose=verbose
        )
        self._hpo(
            task=task,
            ensemble=ensemble,
            models=models,
            n_folds=n_folds_cv,
            random_state=random_state,
            n_jobs=n_jobs,
            n_trials=n_trials,
            model_configs=model_configs
        )
        self._evaluating(
            task=task
        )

    def create_report(self):
        pack_dir = osp.dirname(__file__)
        final_path = osp.join(self.outputdir, 'result_analysis.qmd')
        if self.metadata['trainer-metadata']['task'] == 'class':
            tmp_path = osp.join(pack_dir, 'data', 'result_analysis_c.qmd')
        else:
            tmp_path = osp.join(pack_dir, 'data', 'result_analysis_r.qmd')
        shutil.copyfile(tmp_path, final_path)
        sp.run(['quarto', 'render', final_path])
        os.remove(final_path)

    def _hpo(
        self,
        task: str,
        ensemble: str,
        models: Union[str, List[str]],
        model_configs: Dict[str, dict],
        n_folds: int,
        n_jobs: int,
        n_trials: int,
        random_state: int
    ):
        if task == 'class':
            metric = 'mcc'
        else:
            metric = 'spcc'

        self.metadata['status'] = 'hpo'
        self.metadata['trainer-metadata'] = {
            'models': models,
            'n-folds': n_folds,
            'custom-hpspace': model_configs,
            'random-state': random_state,
            'n-jobs': n_jobs,
            'task': task
        }
        self.save_metadata()
        self.trainer = OptunaTrainer(
            task=task,
            direction='maximize',
            metric=metric,
            ensemble=ensemble
        )
        start = time.time()
        train_x = {rep: val[self.parts['train']]
                   for rep, val in self.x.items()}
        train_y = self.df[self.label_field].to_numpy()[self.parts['train']]
        self.trainer.hpo(
            x=train_x,
            y=train_y,
            models=models,
            n_folds=n_folds,
            n_trials=n_trials,
            patience=n_trials//5,
            custom_hpspace=model_configs,
            random_state=random_state,
            n_jobs=n_jobs,
            db_file=osp.join(self.meta_dir, 'database.sql'),
            study_name=f'apml-{self.hpo_run}'
        )
        self.hpo_run += 1
        end = time.time()
        self.metadata['status'] = 'trained'
        self.metadata['trainer-metadata'].update({
            'execution-time': end - start,
            'best-model': self.trainer.best_config,
            'metric': metric,
            'n-trials': n_trials,
            'patience': n_trials // 5,
            'best-run': int(self.trainer.best_run),
            'hpo-run': self.hpo_run
        })
        input_trial = {rep: self.x[rep][:1]
                       for rep in self.trainer.best_model.reps}
        history_path = osp.join(self.meta_dir, 'hpo_history.tsv')
        self.trainer.history.to_csv(history_path, index=False, sep='\t')
        self.trainer.best_model.predict(input_trial)
        self.trainer.best_model.save(osp.join(self.outputdir, 'ensemble'))
        self.ensemble = self.trainer.best_model
        self.save_metadata()

    def _representing(
        self,
        reps: Union[str, List[str], Dict[str, RepEngineBase]],
        device: str,
        n_jobs: int,
        verbose: bool
    ):
        all_available = True
        for rep in reps:
            if rep not in self.x:
                all_available = False
                break

        if all_available:
            return
        reps = [r for r in reps if r not in self.x]
        prot, mol = False, False

        for rep in reps:
            if rep in PLMs and not prot:
                prot = True
            elif (rep in CLMs or rep in FPs) and not mol:
                mol = True

        if 'to-sequences' in self.metadata['pipeline-1']['name'] and mol:
            self.preprocess_data('to-smiles', n_jobs=n_jobs,
                                 verbose=verbose)
        if 'to-smiles' in self.metadata['pipeline-1']['name'] and prot:
            self.preprocess_data('to-sequences', n_jobs=n_jobs,
                                 verbose=verbose)

        self.metadata['status'] = 'representing'
        self.save_metadata()

        if isinstance(reps, dict):
            for name, repengine in reps.items():
                self.execution[name] = {'start': time.time()}
                self.x[name] = repengine.compute_reps(
                    self.df[self.sequence_field], verbose=verbose, batch_size=16
                )
                self.execution[name]['end'] = time.time()
            reps = list(reps.keys())
        else:
            for rep in reps:
                if verbose:
                    print(f"Computing {rep} representations...")

                if rep in PLMs or rep in CLMs:
                    from autopeptideml.reps.lms import RepEngineLM
                    self.execution[rep] = {'start': time.time()}

                    repengine = RepEngineLM(rep, average_pooling=True,
                                            fp16=True)
                    repengine.move_to_device(device)

                elif rep.split('-')[0] in FPs:
                    from autopeptideml.reps.fps import RepEngineFP
                    if len(rep.split('-')) == 1:
                        rep = f'{rep}-8-2048'
                    elif len(rep.split('-')) == 2:
                        rep = f"{rep.split('-')[0]}-{rep.split('-')[1]}-2048"
                    self.execution[rep] = {'start': time.time()}
                    repengine = RepEngineFP(
                        rep=rep.split('-')[0],
                        radius=int(rep.split('-')[1]),
                        nbits=int(rep.split('-')[2])
                    )
                elif rep == 'one-hot':
                    from autopeptideml.reps.seq_based import RepEngineOnehot
                    self.execution[rep] = {'start': time.time()}

                    repengine = RepEngineOnehot(max_length=50)

                batch_size = 128 if repengine.get_num_params() < 2e7 else 16
                if rep in PLMs or rep == 'one-hot':
                    if 'to-sequences' in self.metadata['pipeline-1']['name']:
                        self.execution[rep] = {'start': time.time()}

                        self.x[rep] = repengine.compute_reps(
                            self.df[f'{self.sequence_field}'], verbose=verbose,
                            batch_size=batch_size
                        )
                    else:
                        self.execution[rep] = {'start': time.time()}

                        self.x[rep] = repengine.compute_reps(
                            self.df[f'{self.sequence_field}-2'], verbose=verbose,
                            batch_size=batch_size
                        )
                elif rep in CLMs or rep.split('-')[0] in FPs:
                    if 'to-smiles' in self.metadata['pipeline-1']['name']:
                        self.execution[rep] = {'start': time.time()}

                        self.x[rep] = repengine.compute_reps(
                            self.df[f'{self.sequence_field}'], verbose=verbose,
                            batch_size=batch_size
                        )
                    else:
                        self.execution[rep] = {'start': time.time()}

                        self.x[rep] = repengine.compute_reps(
                            self.df[f'{self.sequence_field}-2'],
                            verbose=verbose,
                            batch_size=batch_size
                        )
                self.execution[rep]['end'] = time.time()

        self.x.update({rep: np.array(value) for rep, value in self.x.items()})
        path = osp.join(self.meta_dir, 'reps.pckl')
        pickle.dump(self.x, open(path, 'wb'))
        self.metadata['status'] = 'represented'
        if 'reps-metadata' in self.metadata:
            self.metadata['reps-metadata'].update({'reps': list(self.x.keys())})
        else:
            self.metadata['reps-metadata'] = {'reps':  list(self.x.keys())}

        self.metadata['reps-metadata'].update({
            f'{rep}-execution-time':
            self.execution[rep]['end'] - self.execution[rep]['start']
            for rep in self.x.keys()})
        self.save_metadata()

    def _partitioning(
        self,
        split_strategy: str,
        hestia_generator: HestiaGenerator,
        verbose: bool,
        n_jobs: int,
        random_state: int,
        partitions: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        if verbose:
            print("Partitioning...")

        if partitions is not None:
            part_path = osp.join(self.outputdir, 'parts.pckl')
            parts = {k: v for k, v in partitions.items()}
            pickle.dump(parts, open(part_path, 'wb'))
            self.parts = partitions
            return
        if self.parts is not None:
            return

        SPLIT_STRATEGIES = ['random', 'min', 'good', None]
        self.metadata['status'] = 'partitioning'
        self.save_metadata()

        start = time.time()
        if split_strategy not in SPLIT_STRATEGIES:
            raise ValueError(f"split_strategy: {split_strategy} is not valid.",
                             F"Please choose: {', '.join(SPLIT_STRATEGIES)}")
        if hestia_generator is not None:
            sim_args = hestia_generator.sim_args.to_dict()
            min_part, _ = hestia_generator.get_partition('min', filter=0.185)
        elif split_strategy == 'random':
            self.parts = {
                'test': self.df.sample(frac=0.2, random_state=random_state).index.to_numpy()
            }
            self.parts['train'] = self.df[~self.df.index.isin(self.parts['test'])].index.to_numpy()
            sim_args = {}
            min_part = 'NA'
        elif 'to-smiles' in self.metadata['pipeline-1']['name']:
            sim_args = SimArguments(
                data_type='small molecule',
                fingerprint='mapc' if len(self.df) < 10_000 else 'ecfp',
                sim_function='jaccard' if len(self.df) < 10_000 else 'tanimoto',
                bits=2048,
                radius=4,
                field_name=self.sequence_field,
                min_threshold=0.1,
                threads=n_jobs,
                verbose=3 if verbose else 0,
            )
            hdg = HestiaGenerator(self.df, verbose=verbose)
            hdg.calculate_partitions(sim_args, label_name=self.label_field,
                                     min_threshold=0.2, threshold_step=0.1,
                                     valid_size=0., random_state=random_state)
            sim_args = sim_args.to_dict()
            if split_strategy == 'min':
                min_part, self.parts = hdg.get_partition('min', filter=0.185)
            elif split_strategy == 'good':
                self.parts = hdg.get_partitions(filter=0.185, return_dict=True)
                min_part, _ = hdg.get_partition('min', filter=0.185)
        elif 'to-sequences' in self.metadata['pipeline-1']['name']:
            sim_args = SimArguments(
                data_type='sequence',
                field_name=self.sequence_field,
                min_threshold=0.1,
                alignment_algorithm='mmseqs',
                prefilter=True,
                denominator='shortest',
                threads=n_jobs,
                verbose=verbose
            )
            hdg = HestiaGenerator(self.df, verbose=verbose)
            hdg.calculate_partitions(sim_args, label_name=self.label_field,
                                     min_threshold=0.2, threshold_step=0.1,
                                     valid_size=0., random_state=random_state)
            sim_args = sim_args.to_dict()
            if split_strategy == 'min':
                th, self.parts = hdg.get_partition('min', filter=0.185)
            elif split_strategy == 'good':
                self.parts = hdg.get_partitions(filter=0.185, return_dict=True)

        part_path = osp.join(self.meta_dir, 'parts.pckl')
        self.parts = {k: v for k, v in self.parts.items() if k != 'clusters'}
        pickle.dump(self.parts, open(part_path, 'wb'))
        end = time.time()
        self.metadata['partitioning-metadata'] = {
            'execution-time': end - start,
            'split-strategy': split_strategy,
            'sim-args': sim_args,
            'number-partitions': len(self.parts),
            'n-jobs': n_jobs,
            'random-state': random_state,
            'hestia-version': hestia_version,
            'min-part': min_part,
            'train-size': len(self.parts['train']),
            'test-size': len(self.parts['test'])
        }
        self.metadata['status'] = 'partitioned'
        self.save_metadata()

    def _evaluating(self, task: str):
        self.metadata['status'] = 'evaluating'
        self.save_metadata()

        start = time.time()
        test_x = {rep: val[self.parts['test']]
                  for rep, val in self.x.items()}
        test_y = self.df[self.label_field].to_numpy()[self.parts['test']]
        try:
            preds, _ = self.ensemble.predict_proba(test_x)
            preds = preds[:, 1]
        except AttributeError:
            preds, _ = self.ensemble.predict(test_x)

        result = evaluate(preds, test_y, pred_task=task)
        np.save(open(osp.join(self.meta_dir, 'preds.npy'), 'wb'), preds)
        self.test_result = pd.DataFrame([result])
        end = time.time()
        self.metadata['test-metadata'] = {
            str(metric): float(value) for metric, value in result.items()
        }
        self.metadata['test-metadata']['execution-time'] = end - start
        self.metadata['status'] = 'evaluated'
        self.save_metadata()
