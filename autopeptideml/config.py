import copy
import os
import os.path as osp
import json
import yaml
from typing import *

import pandas as pd
import numpy as np
from hestia import HestiaGenerator, SimArguments

from .pipeline import Pipeline, CanonicalCleaner, CanonicalFilter
from .reps import RepEngineBase
from .train import BaseTrainer, OptunaTrainer, GridTrainer, NoHpoTrainer
from .train.metrics import evaluate
from .db import Database


class AutoPeptideML:
    config: dict
    pipeline: Pipeline
    rep: Dict[str, RepEngineBase]
    train: BaseTrainer
    db: Database
    parts: dict
    x: Optional[Dict[str, np.ndarray]] = None
    y: np.ndarray

    def __init__(self, config: dict):
        self.pipe_config = config['pipeline']
        self.rep_config = config['representation']
        self.train_config = config['train']
        self.db_config = config['databases']
        self.test_config = config['test']

        self.config = config
        self.outputdir = config['outputdir']
        if osp.isdir(self.outputdir):
            raise ValueError(f"Output dir: {self.outputdir} exists.")
        else:
            os.makedirs(self.outputdir)

        if isinstance(self.pipe_config, str):
            pipe_config_path = osp.join(self.pipe_config, 'config.yml')
            self.pipe_config = yaml.safe_load(open(pipe_config_path))['pipeline']
        else:
            self.pipeline = self._load_pipeline(self.pipe_config)

        if 'precalculated' in self.db_config:
            self.get_precalculated_db(self.db_config)
        else:
            self.db = self._load_database(self.db_config)

        if isinstance(self.rep_config, str):
            self.outputdir = self.rep_config
            self.get_precalculated_reps(self.rep_config)
            self.outputdir = self.config['outputdir']
        else:
            self.reps = self._load_representation(self.rep_config)

        if isinstance(self.test_config, str):
            config_path = osp.join(self.test_config, 'config.yml')
            self.test_config = yaml.safe_load(open(config_path))['test']
            self.parts = self._load_test(self.test_config)
        else:
            self.parts = self._load_test(self.test_config)

        if isinstance(self.train_config, str):
            train_config_path = osp.join(self.train_config, 'config.yml')
            self.train_config = yaml.safe_load(open(train_config_path))['train']
            self.train = self._load_trainer(self.train_config)
        else:
            self.train = self._load_trainer(self.train_config)

    def _load_pipeline(self, pipe_config: dict) -> Pipeline:
        elements = []

        for config in pipe_config['elements']:
            name = list(config.keys())[0]
            if 'pipe' in name:
                item = self._load_pipeline(config[name])
            else:
                if 'filter-smiles' in config:
                    from .pipeline.smiles import FilterSMILES
                    config = config['filter-smiles']
                    config = {} if config is None else config
                    item = FilterSMILES(**config)

                elif 'sequence-to-smiles' in config:
                    from .pipeline.smiles import SequenceToSMILES
                    config = config['sequence-to-smiles']
                    config = {} if config is None else config
                    item = SequenceToSMILES(**config)

                elif 'canonical-cleaner' in config:
                    config = config['canonical-cleaner']
                    config = {} if config is None else config
                    item = CanonicalCleaner(**config)

                elif 'canonical-filter' in config:
                    config = config['canonical-filter']
                    config = {} if config is None else config
                    item = CanonicalFilter(**config)

            elements.append(item)
        return Pipeline(name=pipe_config['name'], elements=elements,
                        aggregate=pipe_config['aggregate'])

    def _load_representation(self, rep_config: str) -> Dict[str, RepEngineBase]:
        out = {}
        for r in rep_config['elements']:
            name = list(r.keys())[0]
            r_config = r[name]
            if 'lm' in r_config['engine'].lower():
                from .reps.lms import RepEngineLM
                re = RepEngineLM(r_config['model'], r_config['average_pooling'])
                re.batch_size = r_config['batch_size']
                re.device = r_config['device']

            elif 'fp' in r_config['engine'].lower():
                from .reps.fps import RepEngineFP
                re = RepEngineFP(r_config['fp'], nbits=r_config['nbits'],
                                 radius=r_config['radius'])

            elif 'onehot' in r_config['engine'].lower():
                from .reps import RepEngineOnehot
                re = RepEngineOnehot(rep_config['max_length'])

            out[name] = re
        return out

    def _load_trainer(self, train_config: dict) -> BaseTrainer:
        hspace = train_config['hspace']
        optim_strategy = train_config['optim_strategy']
        if optim_strategy['trainer'] == 'optuna':
            trainer = OptunaTrainer(hspace, optim_strategy)
        elif optim_strategy['trainer'] == 'grid':
            trainer = GridTrainer(hpspace=hspace, optim_strategy=optim_strategy)
        return trainer

    def _load_database(self, db_config: dict) -> Database:
        db = Database(
            db_config['dataset']['path'], pipe=self.pipeline,
            feat_fields=db_config['dataset']['feat_fields'],
            label_field=db_config['dataset']['label_field'],
            verbose=db_config['dataset']['verbose']
        )
        if 'neg_database' in db_config:
            db2 = Database(
                db_config['neg_database']['path'], pipe=self.pipeline,
                feat_fields=db_config['neg_database']['feat_fields'],
                verbose=db_config['neg_database']['verbose']
            )
            db.df = db.df[db.df['Y'] == 1].copy()
            db.add_negatives(
                db2,
                columns_to_exclude=db_config['neg_database']['columns_to_exclude']
            )
        return db

    def _load_test(self, test_config: dict) -> HestiaGenerator:
        parts_path = osp.join(self.outputdir, 'parts.pckl')
        if osp.exists(parts_path):
            hdg = HestiaGenerator(self.db.df, verbose=test_config['verbose'])
            hdg.from_precalculated(parts_path)
        else:
            sim_args = SimArguments(**test_config['sim_arguments'])
            hdg = HestiaGenerator(self.db.df, verbose=test_config['verbose'])
            hdg.calculate_partitions(sim_args, label_name=self.db.label_field,
                                     threshold_step=test_config['threshold_step'],
                                     min_threshold=test_config['min_threshold'],
                                     partition_algorithm=test_config['algorithm'],
                                     valid_size=0)
        self.hdg = hdg
        partitions = hdg.get_partitions(filter=test_config['filter'],
                                        return_dict=True)
        return partitions

    def _get_folds(self, val_config: dict, part, y) -> Dict[str, np.ndarray]:
        if val_config['type'] == 'kfold':
            from sklearn.model_selection import StratifiedKFold
            kf = StratifiedKFold(n_splits=val_config['k'], shuffle=True,
                                 random_state=val_config['random_state'])
            folds = [fold for fold in kf.split(part['train'], y[part['train']])]
        elif val_config['type'] == 'single':
            from sklearn.model_selection import train_test_split
            folds = train_test_split(part['train'], test_size=val_config['size'],
                                     random_state=val_config['random_state'],
                                     shuffle=True)
        else:
            raise NotImplementedError(f"Validation split: {val_config['type']} is not supported.")
        return folds

    def _get_reps(self) -> Dict[str, np.ndarray]:
        if self.x is not None:
            return self.x
        x = {}
        rep_dir = osp.join(self.outputdir, 'reps')
        os.makedirs(rep_dir, exist_ok=True)

        for name, rep in self.reps.items():
            path = osp.join(rep_dir, f'{rep.name}.pckl')
            if osp.exists(path):
                x[name] = np.load(path, allow_pickle=True)
            else:
                if self.rep_config['verbose']:
                    print(f'Computing representations with: {name}')
                if 'lm' in rep.name:
                    rep.move_to_device(rep.device)
                    x[name] = rep.compute_reps(
                        self.db.df[self.db.feat_fields[0]], verbose=True,
                        batch_size=rep.batch_size
                    )
                else:
                    x[name] = rep.compute_reps(self.db.df[self.db.feat_fields[0]],
                                               verbose=True)
                x[name].dump(path)
        self.x = x
        return x

    def save_models(
        self,
        ensemble_path: str,
        backend: str = 'onnx'
    ):
        if backend == 'joblib':
            try:
                import joblib
            except ImportError:
                raise ImportError(
                    'This backend requires joblib.',
                    'Please try: `pip install joblib`'
                )
        elif backend == 'onnx':
            try:
                import onnxmltools as onxt
                from skl2onnx.common.data_types import FloatTensorType
                from skl2onnx import to_onnx
            except ImportError:
                raise ImportError(
                    'This backend requires onnx.',
                    'Please try: `pip install onnxmltools skl2onnx`'
                )
        else:
            raise NotImplementedError(f"Backend: {backend} not implemented.",
                                      "Please try: `onnx` or `joblib`.")
        for th, model in self.models.items():
            model['save_path'] = []
            if backend == 'onnx':
                for idx, clf in enumerate(model['models']):
                    m_x = self.x[model['reps'][idx]]
                    variable_type = FloatTensorType([None, m_x.shape[1]])
                    if 'LGBM' in str(clf):
                        clf_onx = onxt.convert_lightgbm(
                            clf,
                            initial_types=[('float_input', variable_type)]
                        )
                    elif 'XGB' in str(clf):
                        clf_onx = onxt.convert_xgboost(
                            clf,
                            initial_types=[('float_input', variable_type)]
                        )
                    else:
                        clf_onx = to_onnx(clf, [('X', variable_type)])

                    if 'class' in str(clf).lower():
                        name = f'{th}_{idx}_class.onnx'
                    else:
                        name = f'{th}_{idx}_reg.onnx'
                    model['save_path'].append(name)

                    with open(osp.join(ensemble_path, name), "wb") as f:
                        f.write(clf_onx.SerializeToString())
            else:
                for idx, clf in enumerate(self.models['models']):
                    if 'class' in str(clf).lower():
                        name = f'{idx}_class.onnx'
                    else:
                        name = f'{idx}_reg.onnx'
                    model['save_path'].append(name)
                    joblib.dump(clf, open(osp.join(ensemble_path, name)), 'wb')

    def run_hpo(self):
        self._get_reps()
        x = self.x
        y = self.db.df[self.db.label_field].to_numpy()
        models = {}

        if self.train_config['optim_strategy']['partition'] == 'all':
            for th, part in self.parts:
                folds = self._get_folds(self.config['val'], part, y)
                self.train.hpo(folds, x, y)
                models[th] = self.train.best_model
        else:
            part = self.parts[self.train_config['optim_strategy']['partition']]
            folds = self._get_folds(self.config['val'], part, y)
            self.train.hpo(folds, x, y)
            trainer2 = NoHpoTrainer(self.train.best_config,
                                    self.train.optim_strategy)
            for th, part in self.parts.items():
                folds = self._get_folds(self.config['val'], part, y)
                model = trainer2.train(folds, x, y)
                models[th] = model
        self.models = models
        return models

    def run_evaluation(self, models) -> pd.DataFrame:
        self._get_reps()
        x = self.x
        y = self.db.df[self.db.label_field].to_numpy()

        results = []
        if self.test_config['partitions'] == 'all':
            parts = self.parts
        elif self.test_config['partitions'] == 'min':
            th = min([th for th, v in self.parts.items() if th != 'random'])
            parts = {th: self.parts[th]}
        elif self.test_config['partitions'] == 'max':
            th = max([th for th, v in self.parts.items() if th != 'random'])
            parts = {th: self.parts[th]}
        elif isinstance(self.test_config['partitions'], float):
            parts = {th: self.parts[self.test_config['partitions']]}
        elif self.test_config['partitions'] == 'random':
            parts = parts['random']
        else:
            raise ValueError(f"Test partitions: {self.test_config['partitions']} are not supported.")
        for th, part in parts.items():
            test_y = y[part['test']]
            preds = np.zeros(test_y.shape)
            ensemble = models[th]
            for arch, rep in zip(ensemble['models'], ensemble['reps']):
                test_x = x[rep][part['test']]
                preds += (arch.predict_proba(test_x)[:, 1] /
                          len(ensemble['models']))
            result = evaluate(preds, test_y, self.train.optim_strategy['task'])
            result['threshold'] = th
            results.append(result)

        df = pd.DataFrame(results)
        path = osp.join(self.outputdir, 'results.csv')
        df.to_csv(path, index=False)
        return df

    def save_experiment(self, model_backend: str = 'onnx', save_reps: bool = False,
                        save_test: bool = True):
        config_path = osp.join(self.outputdir, 'config.yml')
        parts_path = osp.join(self.outputdir, 'parts.pckl')
        model_info = osp.join(self.outputdir, 'model_info')
        ensemble_path = osp.join(model_info, 'ensemble')
        reps_dir = osp.join(self.outputdir, 'reps')

        ensemble_config_path = osp.join(model_info, 'ensemble_config.json')

        os.makedirs(ensemble_path)
        os.makedirs(reps_dir, exist_ok=True)

        self.save_models(ensemble_path, model_backend)
        self.save_database()
        to_save = copy.deepcopy(self.models)
        for th, model in to_save.items():
            del model['models']

        if save_test:
            self.hdg.save_precalculated(parts_path)
        if save_reps:
            self.save_reps(reps_dir)

        self.config['databases'] = self.db_config
        self.config['representation'] = self.rep_config
        self.config['test'] = self.test_config
        self.config['train'] = self.train_config
        yaml.safe_dump(self.config, open(config_path, 'w'))
        json.dump(to_save, open(ensemble_config_path, 'w'))

    def save_database(self):
        db_path = osp.join(self.outputdir, 'db.csv')
        self.db.df.to_csv(db_path, index=False)

    def save_reps(self, rep_dir: str):
        for name, rep in self.reps.items():
            path = osp.join(rep_dir, f'{rep.name}.pckl')
            self.x[name].dump(path)

    def get_precalculated_reps(self, outputdir: str) -> Dict[str, np.ndarray]:
        config_path = osp.join(outputdir, 'config.yml')
        config = yaml.safe_load(open(config_path))
        rep_config = config['representation']
        self.rep_config = rep_config
        self.reps = self._load_representation(rep_config)
        self.x = self._get_reps()
        return self.x

    def get_precalculated_test(self, outputdir: str) -> Dict[str, np.ndarray]:
        config_path = osp.join(outputdir, 'config.yml')
        config = yaml.safe_load(open(config_path))
        rep_config = config['test']
        c_outputdir = self.outputdir
        self.outputdir = config['outputdir']
        self.parts = self._load_test(rep_config)
        self.outputdir = c_outputdir
        return self.parts

    def get_precalculated_db(self, db_config: dict) -> Database:
        db_path = osp.join(db_config['precalculated'], 'db.csv')
        self.db = Database(db_path, feat_fields=db_config['feat_fields'],
                           label_field=db_config['label_field'])


if __name__ == "__main__":
    # import shutil
    # path = 'AutoPeptideML/autopeptideml/data/configs/optuna_train.yaml'
    # try:
    #     config = yaml.safe_load(open(path))
    # except yaml.scanner.ScannerError:
    #     raise RuntimeError("The YAML config file has syntax errors.")
    # if osp.exists(config['outputdir']):
    #     shutil.rmtree(config['outputdir'])
    path = 'AutoPeptideML/autopeptideml/data/configs/restart.yml'
    config = yaml.safe_load(open(path))
    apml = AutoPeptideML(config)
    models = apml.run_hpo()
    r_df = apml.run_evaluation(models)
    apml.save_experiment(save_reps=True, save_test=False)
    print(r_df)
