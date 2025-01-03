import copy
import json
import os

import joblib
import numpy as np
import pandas as pd

from ..data.metrics import METRIC2FUNCTION, METRICS
from ..data.algorithms import SUPPORTED_MODELS, SYNONYMS


NO_N_JOBS = []
NO_N_JOBS.extend(SYNONYMS['svm'])
NO_N_JOBS.extend(SYNONYMS['mlp'])
NO_N_JOBS.extend(SYNONYMS['xgboost'])

PROBABILITY = []
PROBABILITY.extend(SYNONYMS['svm'])


class FlexibleObjective:
    def __init__(
        self,
        config,
        train_df: pd.DataFrame,
        folds: list,
        id2rep: dict,
        threads: int,
        outputdir: str
    ):
        self.best_model = None
        self.model = None
        self.outputdir = outputdir
        self.cross_val, self.folds = self.generate_folds(
            train_df, folds, id2rep
        )
        self.threads = threads
        self.name = config['model']
        self.config = config['hyperparameter-space']

    def __call__(self, trial):
        hyperparameter_space = {}

        for variable in self.config:
            if variable['type'] == 'int':
                hyperparameter_space[variable['name']] = trial.suggest_int(f"{self.name}_{variable['name']}", variable['min'], variable['max'], log=variable['log'])

            elif variable['type'] == 'float':
                hyperparameter_space[variable['name']] = trial.suggest_float(f"{self.name}_{variable['name']}", variable['min'], variable['max'], log=variable['log'])

            elif variable['type'] == 'categorical':
                hyperparameter_space[variable['name']] = trial.suggest_categorical(f"{self.name}_{variable['name']}", variable['values'])

            elif variable['type'] == 'fixed':
                hyperparameter_space[variable['name']] = variable['value']

        if self.name.lower() in PROBABILITY:
            hyperparameter_space['probability'] = True

        if self.name.lower() not in NO_N_JOBS:
            hyperparameter_space['n_jobs'] = self.threads

        classifier_obj = self.get_model(self.name)(**hyperparameter_space)
        cross_validation_result = self.cross_validate(
            classifier_obj=classifier_obj,
            x=self.cross_val['x'],
            y=self.cross_val['y'],
            scoring=METRICS,
            cv=self.folds
        )
        optimization_metric = cross_validation_result['matthews_corrcoef'].mean()
        self.model = cross_validation_result
        return optimization_metric

    def generate_folds(
        self,
        train_df: pd.DataFrame,
        folds: list,
        id2rep: dict
    ):
        cross_val = {}
        fold_reps = []

        x_train = np.array([id2rep[id] for id in train_df.id]).reshape(len(train_df), -1)

        cross_val['x'] = {id: embd for id, embd in zip(train_df.id.tolist(), x_train)}
        cross_val['y'] = {id: y for id, y in zip(train_df.id.tolist(), train_df.Y.tolist())}

        for fold in folds:
            fold_reps.append((fold['train'].id.tolist(), fold['val'].id.tolist()))
        return cross_val, fold_reps

    def cross_validate(self, classifier_obj, x, y, scoring, cv):
        result = {'estimators': []}
        for idx, (train_idx, val_idx) in enumerate(cv):
            train_x = np.array([x[i] for i in train_idx])
            train_y = np.array([y[i] for i in train_idx])
            val_x = np.array([x[i] for i in val_idx])
            val_y = np.array([y[i] for i in val_idx])
            clf = copy.deepcopy(classifier_obj)
            clf.fit(train_x, train_y)
            preds = clf.predict(val_x)

            for metric in scoring:
                if metric not in result:
                    result[metric] = []
                if metric in ['f1', 'recall', 'precision', 
                              'f1_weighted', 'recall_weighted',
                              'precision_weighted']:
                    result[metric].append(METRIC2FUNCTION[metric](val_y, preds, zero_division=0))
                else:
                    result[metric].append(METRIC2FUNCTION[metric](val_y, preds))
            result[f'estimators'].append(clf)

        for metric in scoring:
            result[metric] = np.array(result[metric])
        return result

    def callback(self, study, trial):
        if study.best_trial == trial:
            self.best_model = self.model
            # for idx, estimator in enumerate(self.best_model['estimators']):
                # joblib.dump(estimator, open(os.path.join(self.outputdir, 'ensemble', f'{self.name}_{idx}.joblib'), 'wb'))
            json.dump(self.best_model['estimators'][0].get_params(),
                      open(os.path.join(self.outputdir, 'best_configs', f'{self.name}.json'), 'w'), indent=4)

    def get_model(self, model: str):
        for key in SUPPORTED_MODELS:
            if model.lower() in SYNONYMS[key]:
                return SUPPORTED_MODELS[key]
        raise ValueError(f'Model {model} is not supported. Please check for any misspellings and, if appropriate, raise an issue in the Project GitHub Repository.')


class ModelSelectionObjective(FlexibleObjective):
    def __init__(
        self,
        config,
        train_df: pd.DataFrame,
        folds: list,
        id2rep: dict,
        threads: int,
        outputdir: str
    ):
        self.best_model = None
        self.model = None
        self.outputdir = outputdir
        self.cross_val, self.folds = self.generate_folds(
            train_df, folds, id2rep
        )
        self.threads = threads
        self.config = config

    def __call__(self, trial):
        model2idx = {x['model']: idx for idx, x in enumerate(self.config)}
        model = trial.suggest_categorical('model', [x['model'] for x in self.config])
        hyperparameter_space = {}
        for variable in self.config[model2idx[model]]['hyperparameter-space']:
            if variable['type'] == 'int':
                hyperparameter_space[variable['name']] = trial.suggest_int(f"{model}_{variable['name']}", variable['min'], variable['max'], log=variable['log'])

            elif variable['type'] == 'float':
                hyperparameter_space[variable['name']] = trial.suggest_float(f"{model}_{variable['name']}", variable['min'], variable['max'], log=variable['log'])

            elif variable['type'] == 'categorical':
                hyperparameter_space[variable['name']] = trial.suggest_categorical(f"{model}_{variable['name']}", variable['values'])

            elif variable['type'] == 'fixed':
                hyperparameter_space[variable['name']] = variable['value']

        if self.name.lower() in NO_N_JOBS:
            hyperparameter_space['probability'] = True
        else:
            hyperparameter_space['n_jobs'] = self.threads

        classifier_obj = self.get_model(model)(**hyperparameter_space)
        cross_validation_result = self.cross_validate(
            classifier_obj=classifier_obj,
            x=self.cross_val['x'],
            y=self.cross_val['y'],
            scoring=METRICS,
            cv=self.folds
        )
        optimization_metric = cross_validation_result['matthews_corrcoef'].mean()
        self.model = cross_validation_result
        self.model['name'] = model
        return optimization_metric

    def callback(self, study, trial):
        if study.best_trial == trial:
            self.best_model = self.model
            # for idx, estimator in enumerate(self.best_model['estimators']):
            #     joblib.dump(estimator, open(os.path.join(self.outputdir, 'ensemble', f"{self.model['name']}_{idx}.joblib"), 'wb'))
            json.dump(self.best_model['estimators'][0].get_params(), open(os.path.join(self.outputdir, 'best_configs', f"{self.model['name']}.json"), 'w'), indent=4)


class UniDL4BioPep_Objective(FlexibleObjective):
    def __call__(self, trial):
        optimizer = {}
        # criteria = {}
        
        for variable in self.config['optimizer']:
            if variable['type'] == 'int':
                optimizer[variable['name']] = trial.suggest_int(f"{self.name}_{variable['name']}", variable['min'], variable['max'], log=variable['log'])
            elif variable['type'] == 'float':
                optimizer[variable['name']] = trial.suggest_float(f"{self.name}_{variable['name']}", variable['min'], variable['max'], log=variable['log'])
            elif variable['type'] == 'categorical':
                optimizer[variable['name']] = trial.suggest_categorical(f"{self.name}_{variable['name']}", variable['values'])
        # for variable in self.config['criteria']:
        #     if variable['type'] == 'int':
        #         criteria[variable['name']] = trial.suggest_int(f"{self.name}_{variable['name']}", variable['min'], variable['max'], log=variable['log'])
        #     elif variable['type'] == 'float':
        #         criteria[variable['name']] = trial.suggest_float(f"{self.name}_{variable['name']}", variable['min'], variable['max'], log=variable['log'])
        #     elif variable['type'] == 'categorical':
        #         criteria[variable['name']] = trial.suggest_categorical(f"{self.name}_{variable['name']}", variable['values'])

        classifier_obj = self.get_model(self.name)(
            optimizer=optimizer,
            epochs=self.config['epochs'],
            # criteria=criteria,
            criteria=None,
            logger=os.path.join(self.outputdir, 'logger'))

        cross_validation_result = self.cross_validate(
            classifier_obj=classifier_obj,
            x=self.cross_val['x'],
            y=self.cross_val['y'],
            scoring=METRICS,
            cv=self.folds
        )
        optimization_metric = cross_validation_result['matthews_corrcoef'].mean()
        best_model = np.argmax(cross_validation_result['matthews_corrcoef'])
        cross_validation_result['model'] = cross_validation_result['model'][best_model]
        self.model = cross_validation_result
        return optimization_metric

    def callback(self, study, trial):
        if study.best_trial == trial:
            self.best_model = self.model['model']
            json.dump(self.best_model.get_params(), open(os.path.join(self.outputdir, 'best_configs', f'{self.name}.json'), 'w'), indent=4)

    def cross_validate(self, classifier_obj, x, y, scoring, cv):
        from tqdm import tqdm
        result = {'model': []}
        result.update({metric: [] for metric in scoring})
        for idx, (train_idx, val_idx) in tqdm(enumerate(cv)):
            train_x = np.array([x[i] for i in train_idx])
            train_y = np.array([y[i] for i in train_idx])
            val_x = np.array([x[i] for i in val_idx])
            val_y = np.array([y[i] for i in val_idx])
            clf = copy.deepcopy(classifier_obj)
            clf.fit(train_x, train_y, val_x, val_y)
            report = clf.evaluate(val_x, val_y)
            result[f'model'].append(clf)
            for metric in scoring:
                result[metric].append(report[metric])

        for metric in scoring:
            result[metric] = np.array(result[metric])

        return result
