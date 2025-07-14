import copy
import json
import operator
import os.path as osp
import yaml
import warnings

from typing import *

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from .architectures import *
from .metrics import evaluate
from ..utils import format_numbers, discretizer


PROBABILITY = ['svm']


class EarlyStoppingCallback(object):
    """Early stopping callback for Optuna."""
    def __init__(self, early_stopping_rounds: int,
                 direction: str = "minimize") -> None:
        self.early_stopping_rounds = early_stopping_rounds

        self._iter = 0

        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study, trial) -> None:
        """Do early stopping."""
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            study.stop()


def choose_hps(variable, trial, hpspace, model, key):
    if variable['type'] == 'int':
        hpspace[key] = trial.suggest_int(
            f"{model}_{key}",
            variable['min'],
            variable['max'],
            log=variable['log']
        )

    elif variable['type'] == 'float':
        hpspace[key] = trial.suggest_float(
            f"{model}_{key}",
            variable['min'],
            variable['max'],
            log=variable['log']
        )

    elif variable['type'] == 'categorical':
        hpspace[key] = trial.suggest_categorical(
            f"{model}_{key}",
            variable['values']
        )

    elif variable['type'] == 'fixed':
        hpspace[key] = variable['value']

    return hpspace


class BaseTrainer:
    name: str
    task: str
    metric: str
    direction: str
    ensemble: bool
    random_state: int = 1
    n_folds: int = None
    train_val_ratio: float = None

    def __init__(self, task: str, direction: str = 'maximize',
                 metric: str = None, ensemble: bool = False):
        self.task = task
        self.direction = direction
        if metric is None:
            if task == 'class':
                metric = 'mcc'
            elif task == 'multiclass':
                metric = 'f1_weighted'
            elif task == 'reg':
                metric = 'mse'
            else:
                raise ValueError("Task is not valid. Please try: class, reg, or multiclass")
        self.metric = metric
        self.ensemble = ensemble

        # self.models = models
        # self.hpspace = {
        #     'models': {
        #         'elements': {
        #             **hpspace[model] if model in hpspace
        #                 else get_hpspace(model)
        #             for model in models
        #         }
        #     }
        # }
        # self.hpspace = hpspace
        # self.optim_strategy = optim_strategy
        # self.properties = copy.deepcopy(self.__dict__)
        # self.models = self._import_models(
        #     optim_strategy['task'],
        #     hpspace['models']['elements'].keys()
        # )

    def _define_folds(self, x: np.ndarray, y: np.ndarray,
                      n_folds: int, train_val_ratio: float) -> List[tuple]:
        if train_val_ratio is not None:
            self.train_val_ratio = train_val_ratio
            kf = StratifiedKFold(n_splits=1/train_val_ratio,
                                 random_state=self.random_state,
                                 shuffle=True)
            if 'class' in self.task:
                return [(train, valid) for train, valid in kf.split(x, y)][0]
            elif self.task == 'reg':
                y = discretizer(y)
                return [(train, valid) for train, valid in kf.split(x, y)][0]
        else:
            self.n_folds = n_folds
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                 random_state=self.random_state)
            if 'class' in self.task:
                return [(train, valid) for train, valid in kf.split(x, y)]
            elif self.task == 'reg':
                y = discretizer(y)
                return [(train, valid) for train, valid in kf.split(x, y)]

    def _import_models(
        self, task: str,
        models: List[str]
    ) -> Dict[str, Callable]:
        """
        Imports and initializes model architectures based on the specified task and model list.

        :type task: str
            :param task: The task type (e.g., classification, regression).

        :type models: List[str]
            :param models: List of model names to be imported.

        :rtype: Dict[str, Callable]
            :return: A dictionary mapping model names to their corresponding callable objects.

        :raises NotImplementedError: If a specified model is not implemented.
        """
        archs = {}
        for model in models:
            if model in archs:
                continue
            if model in SKLEARN_MODELS:
                archs.update(load_sklearn_models(task))
            elif model == 'xgboost':
                archs.update(load_xgboost(task))
            elif model == 'lightgbm':
                archs.update(load_lightgbm(task))
            elif model == 'catboost':
                archs.update(load_catboost(task))
            elif model == 'cnn':
                archs.update(load_torch(task))
            else:
                raise NotImplementedError(
                    f"Model: {model} not implemented."
                )
        return archs

    def add_custom_model(self, name: str, model: Callable):
        """
        Introduces a custom learning algorithm to the list of available models of the trainer class.

        :param name: Name of the model, used to define hyperparameter search space.
        :type name: str
        :param model: Callable function or object that initializes a learning algorithm compatible with the scikit-learn API.
        :type model: Callable
        """

    def __str__(self) -> str:
        """
        Returns a JSON string with the properties of the trainer.

        :rtype: str
            :return: A JSON string of the trainer's attributes.
        """
        return json.dumps(self.properties)

    def hpo(
        self,
        train_folds: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Union[Callable, List[Callable]]:
        """
        Abstract method for performing hyperparameter optimization.

        :type train_folds: List[Tuple[np.ndarray, np.ndarray]]
            :param train_folds: Training data folds for cross-validation.

        :rtype: Union[Callable, List[Callable]]
            :return: The best model(s) identified during HPO.

        :raises NotImplementedError: Must be implemented in derived classes.
        """
        raise NotImplementedError


class OptunaTrainer(BaseTrainer):
    """
    Class `OptunaTrainer` implements a hyperparameter optimization (HPO) framework using Optuna.
    It builds on `BaseTrainer` to perform model selection and tuning based on user-defined
    hyperparameter spaces and optimization strategies.

    Attributes:
        :type name: str
        :param name: The name of the trainer. Default is `'optuna'`.

    Example Usage:
        ```python
        trainer = OptunaTrainer(hpspace=my_hpspace, optim_strategy=my_strategy)
        best_model = trainer.hpo(train_folds, x, y)
        ```
    Example Schema for `hpspace`:
        ```python
        hpspace = {
            'models': {
                'type': 'fixed',  # Options: 'fixed', 'ensemble'
                'elements': {
                    'svm': [
                        {'C': {'type': 'float', 'min': 0.1, 'max': 10, 'log': True}},
                        {'kernel': {'type': 'categorical', 'values': ['linear', 'rbf']}},
                        {'probability': {'type': 'fixed', 'value': True}}
                    ],
                    'xgboost': [
                        {'n_estimators': {'type': 'int', 'min': 50, 'max': 200, 'log': False}},
                        {'learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.1, 'log': True}},
                        {'max_depth': {'type': 'int', 'min': 3, 'max': 10, 'log': False}}
                    ]
                }
            },
            'reps': ['representation1', 'representation2']
        }
        ```
    Example Schema for `optim_strategy`:

        ```python
        optim_strategy = {
            'task': 'class',
            'direction': 'maximize',
            'metric': 'mcc',
            'patience': 15,
            'n_steps': 75,
            'n_jobs': 1
        }
        ```
    """
    name = 'optuna'

    def _get_hpspace(self, models: List[str], custom_hpspace: dict) -> dict:
        file_dir = osp.abspath(osp.dirname(__file__))
        config_dir = osp.join(file_dir, '..', 'data', 'h_param_search')
        full_hpspace = {
            'models': {
                'type': 'ensemble' if self.ensemble else 'fixed',
                'elements': {},
                'reps': []
            }
        }
        if models is None:
            models = ALL_MODELS
        for model in models:
            if model in custom_hpspace:
                hpspace = custom_hpspace[model]
            else:
                config_path = osp.join(config_dir, f'{model}_{self.task}.yml')
                hpspace = yaml.safe_load(open(config_path))
            if 'n_jobs' in hpspace:
                hpspace['n_jobs'] = {'type': 'fixed', 'value': self.n_jobs}
            if 'random_state' in hpspace:
                hpspace['random_state'] = {'type': 'fixed',
                                           'value': self.random_state}
            full_hpspace['models']['elements'].update(
                {model: format_numbers(hpspace)}
            )
        return full_hpspace

    def _prepare_hpspace(self, trial) -> dict:
        """Prepares the hyperparameter space for a given Optuna trial.

        :type trial: optuna.trial.Trial
            :param trial: An Optuna trial object used to suggest hyperparameter values.

        :rtype: dict
            :return: A dictionary containing the hyperparameter configurations for the models.

        :raises KeyError: If the hyperparameter space is not properly defined.
        """
        full_hpspace = []
        if self.hpspace['models']['type'] == 'fixed':
            if len(self.hpspace['models']['elements']) == 1:
                m_key = list(self.hpspace['models']['elements'])[0]
            else:
                m_key = trial.suggest_categorical('model', list(self.hpspace['models']['elements'].keys()))
            model = self.hpspace['models']['elements'][m_key]
            hpspace = {}
            for key, variable in model.items():
                if 'condition' in variable:
                    continue

                hpspace = choose_hps(variable, trial, hpspace, model, key)
            for key, variable in model.items():
                if 'condition' not in variable:
                    continue

                hpspace = choose_hps(variable, trial, hpspace, model, key)
                if 'condition' in variable:
                    conditions = variable['condition'].split('|')
                    for condition in conditions:
                        condition = condition.split('-')
                        v, f = condition[0], condition[1]
                        if hpspace[v] != f:
                            del hpspace[v]
                            break

            full_hpspace.append(
                {'name': m_key, 'variables': hpspace,
                 'representation': trial.suggest_categorical(
                  "rep", self.hpspace['reps'])})

        elif self.hpspace['models']['type'] == 'ensemble':
            for m_key, model in self.hpspace['models']['elements'].items():
                hpspace = {}
                for key, variable in model.items():
                    if 'condition' in variable:
                        continue

                    hpspace = choose_hps(variable, trial, hpspace, model, key)
                for key, variable in model.items():
                    if 'condition' not in variable:
                        continue

                    hpspace = choose_hps(variable, trial, hpspace, model, key)
                    if 'condition' in variable:
                        conditions = variable['condition'].split('|')
                        for condition in conditions:
                            condition = condition.split('-')
                            v, f = condition[0], condition[1]
                            if hpspace[v] != f:
                                del hpspace[v]
                                break

                full_hpspace.append(
                    {'name': m_key, 'variables': hpspace,
                     'representation': trial.suggest_categorical(
                        f"{m_key}_rep", self.hpspace['reps'])})

        else:
            models = []
            for m_key, model in self.hpspace['models']['elements'].items():
                models.append(m_key)
            model = trial.suggest_categorical('model', models)
            hpspace = {}

            for key, variable in self.hpspace['models']['elements'][model].items():
                if 'condition' in variable:
                    continue
                hpspace = choose_hps(variable, trial, hpspace, model, key)

            for key, variable in self.hpspace['models']['elements'][model].items():
                if 'condition' not in variable:
                    continue
                hpspace = choose_hps(variable, trial, hpspace, model, key)

                conditions = variable['condition'].split('|')
                for condition in conditions:
                    condition = condition.split('-')
                    v, f = condition[0], condition[1]
                    if hpspace[v] != f:
                        del hpspace[v]
                        break

            full_hpspace.append(
                {'name': model, 'variables': hpspace,
                 'representation': trial.suggest_categorical(
                     f"{model}_rep", self.hpspace['reps']
                 )})
        return full_hpspace

    def _hpo_step(self, trial) -> dict:
        """Executes a single HPO step by evaluating a configuration from the hyperparameter space.

        :type trial: optuna.trial.Trial
            :param trial: An Optuna trial object.

        :rtype: float
            :return: The performance metric for the evaluated configuration.

        :raises ValueError: If the hyperparameter space is improperly defined.
        :raises KeyError: If required fields are missing in the hyperparameter space.
        """
        try:
            hpspace = self._prepare_hpspace(trial)
        except KeyError:
            raise ValueError(
                "Hyperparameter space is not properly defined.",
                "Please check the definition of all fields."
            )
        warnings.filterwarnings('ignore')
        results, supensemble = [], VotingEnsemble([], [])
        train_folds = self.train_folds
        x = self.x
        y = self.y

        for fold_idx, (train_idx, valid_idx) in enumerate(train_folds):
            ensemble = VotingEnsemble([], [])

            for h_m in hpspace:
                arch = self.models[h_m['name']]
                if self.task == 'reg' and h_m['name'] == 'svm':
                    if 'probability' in h_m['variables']:
                        del h_m['variables']['probability']

                arch = arch(**h_m['variables'])
                train_x, train_y = x[h_m['representation']][train_idx], y[train_idx]
                arch.fit(train_x, train_y)
                ensemble.reps.append(h_m['representation'])
                ensemble.models.append(arch)

            valid_y = y[valid_idx]
            try:
                preds, _ = ensemble.predict_proba({k: v[valid_idx]
                                                   for k, v in x.items()})
                preds = preds[:, 1]
            except AttributeError:
                preds, _ = ensemble.predict({k: v[valid_idx]
                                             for k, v in x.items()})

            result = evaluate(preds, valid_y, self.task)
            result.update(h_m)
            result.update({"fold": fold_idx})
            results.append(result)
            supensemble.models.extend(ensemble.models)
            supensemble.reps.extend(ensemble.reps)

        result_df = pd.DataFrame(results)
        if len(self.history) == 0:
            result_df['run'] = 1
        else:
            result_df['run'] = self.history['run'].max() + 1

        self.history = pd.concat([self.history, result_df])

        perf = result_df[self.metric].mean()
        if ((self.direction == 'minimize' and
             perf < self.best_metric) or
           (self.direction == 'maximize' and
           perf > self.best_metric)):
            self.best_metric = perf
            self.best_config = hpspace
            self.best_model = supensemble
        return perf

    def hpo(
        self,
        x: Dict[str, np.ndarray],
        y: np.ndarray,
        models: List[str] = ALL_MODELS,
        n_folds: int = 5,
        train_val_ratio: float = None,
        n_trials: int = 100,
        patience: int = None,
        random_state: int = 1,
        n_jobs: int = 1,
        verbose: int = 2,
        custom_hpspace: dict = {},
        db_file: str = None,
        study_name: str = None
    ) -> Union[Callable, List[Callable]]:
        try:
            import optuna
        except ImportError:
            raise ImportError("This function requires optuna",
                              "Please try: `pip install optuna`")
        if verbose < 1:
            optuna.logging.set_verbosity(optuna.logging.ERROR)
        elif verbose < 3:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Trainer parameters
        self.best_model = None
        self.random_state = random_state
        self.history = pd.DataFrame()
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.models = self._import_models(self.task, models)
        self.hpspace = self._get_hpspace(models, custom_hpspace)
        self.hpspace['reps'] = list(x.keys())
        if patience is None:
            self.patience = n_trials / 5
        self.best_metric = (float("inf") if self.direction == 'minimize'
                            else float('-inf'))

        # Data preparation
        self.train_folds = self._define_folds(x[list(x.keys())[0]], y,
                                              n_folds, train_val_ratio)
        self.x, self.y = x, y

        # Optuna definition
        db_file = f'sqlite:///{db_file}'
        self.study = optuna.create_study(
            direction=self.direction,
            storage=db_file,
            study_name=study_name if study_name is not None else None
        )
        callback = EarlyStoppingCallback(
            early_stopping_rounds=self.patience,
            direction=self.direction
        )
        self.study.optimize(self._hpo_step, n_trials=self.n_trials,
                            callbacks=[callback],
                            gc_after_trial=True, show_progress_bar=verbose == 2)

    @classmethod
    def load_from_db(
        self, path: str, task: str,
        study_name: str
    ) -> "OptunaTrainer":
        try:
            import optuna
        except ImportError:
            raise ImportError("This function requires optuna",
                              "Please try: `pip install optuna`")

        db_file = f'sqlite:///{path}'
        trainer = OptunaTrainer(task=task)
        trainer.study = optuna.load_study(
            storage=db_file, study_name=study_name
        )
        return trainer

# class GridTrainer(BaseTrainer):
#     """
#     Class `GridTrainer` implements a grid search-based hyperparameter optimization (HPO) framework.
#     It systematically explores a predefined hyperparameter space by evaluating all possible combinations.

#     Attributes:
#         :type name: str
#         :param name: The name of the trainer. Default is `'grid'`.

#     Example Usage:
#         ```python
#         trainer = GridTrainer(hpspace=my_hpspace, optim_strategy=my_strategy)
#         best_model = trainer.hpo(train_folds, x, y)
#         ```

#     Example Schema for `hpspace`:

#         ```python
#         hpspace = {
#             'models': {
#                 'type': 'fixed',  # Options: 'fixed', 'ensemble'
#                 'elements': {
#                     'svm': [
#                         {'C': {'type': 'float', 'min': 0.1, 'max': 10, 'steps': 5, 'log': True}},
#                         {'kernel': {'type': 'categorical', 'values': ['linear', 'rbf']}},
#                         {'probability': {'type': 'fixed', 'value': True}}
#                     ],
#                     'xgboost': [
#                         {'n_estimators': {'type': 'int', 'min': 50, 'max': 200, 'steps': 4}},
#                         {'learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.1, 'steps': 5}},
#                         {'max_depth': {'type': 'int', 'min': 3, 'max': 10, 'steps': 3}}
#                     ]
#                 }
#             },
#             'reps': ['representation1', 'representation2']
#         }
#         ```
#     """
#     name = 'grid'

#     def _prepare_hpspace(self) -> dict:
#         """
#         Prepares the hyperparameter space for grid search by generating all possible combinations
#         of hyperparameter values.

#         :rtype: dict
#             :return: A list of dictionaries representing all possible hyperparameter configurations.
#         """
#         full_hpspace = []
#         for m_key, model in self.hpspace['models']['elements'].items():
#             hpspace = {}
#             for key in model:
#                 variable = model[key]

#                 if variable['type'] == 'int':
#                     min, max = variable['min'], variable['max']
#                     if variable['log']:
#                         min, max = np.log10(min), np.log10(max)
#                     step = (max - min) / (variable['steps'] - 1)

#                     hpspace[key] = []
#                     for i in range(variable['steps']):
#                         val = min + step * i
#                         if variable['log']:
#                             hpspace[key].append(int(10**val))
#                         else:
#                             hpspace[key].append(int(val))

#                 elif variable['type'] == 'float':
#                     min, max = variable['min'], variable['max']
#                     if variable['log']:
#                         min, max = np.log10(min), np.log10(max)
#                         step = (max - min) / (variable['steps'] - 1)
#                     hpspace[key] = []
#                     for i in range(variable['steps']):
#                         val = min + step * i
#                         if variable['log']:
#                             hpspace[key].append(10**val)
#                         else:
#                             hpspace[key].append(val)

#                 elif variable['type'] == 'categorical':
#                     hpspace[key] = variable['values']

#                 elif variable['type'] == 'fixed':
#                     hpspace[key] = [variable['value']]

#                 if key.lower() in PROBABILITY:
#                     hpspace['probability'] = [True]
#                 if key.lower() in PROBABILITY and self.optim_strategy['task'] == 'reg':
#                     del hpspace['probability']

#             full_hpspace.append(
#                 {'name': m_key, 'variables': hpspace,
#                  'representation': self.hpspace['reps']})

#         real_hpspace = []
#         for m_space in full_hpspace:
#             model = m_space['name']
#             variables = [v for v in m_space['variables'].values()]
#             names = list(m_space['variables'].keys())
#             names.append('representation')
#             combs = it.product(*variables, m_space['representation'])
#             hpspace = [{n: v[idx] for idx, n in enumerate(names)}
#                       for v in combs]
#             for idx in range(len(hpspace)):
#                 hpspace[idx]['name'] = model
#             real_hpspace.extend(hpspace)
#         return real_hpspace

#     def hpo(
#         self,
#         train_folds: List[Tuple[np.ndarray, np.ndarray]],
#         x: Dict[str, np.ndarray],
#         y: np.ndarray
#     ) -> Union[Callable, List[Callable]]:
#         """Performs hyperparameter optimization using grid search.

#         :type train_folds: List[Tuple[np.ndarray, np.ndarray]]
#             :param train_folds: A list of training folds for cross-validation.

#         :type x: Dict[str, np.ndarray]
#             :param x: A dictionary mapping representations to feature matrices.

#         :type y: np.ndarray
#             :param y: The target labels for the dataset.

#         :rtype: Union[Callable, List[Callable]]
#             :return: The best-performing model(s) after optimization.
#         """
#         self.best_model = None
#         self.best_metric = (
#             float("inf") if self.optim_strategy['direction'] == 'minimize'
#             else float('-inf')
#         )
#         self.metric = self.optim_strategy['metric']
#         hpspace = self._prepare_hpspace()
#         if (self.hpspace['models']['type'] == 'fixed' or
#            self.hpspace['models']['type'] == 'ensemble'):
#             o_hpspace = copy.deepcopy(hpspace)
#             n_hpspace = {}
#             for e in o_hpspace:
#                 if e['name'] in n_hpspace:
#                     n_hpspace[e['name']].append(e)
#                 else:
#                     n_hpspace[e['name']] = [e]

#             hpspace = list(it.product(*list(n_hpspace.values())))

#         pbar = tqdm(hpspace)
#         for idx, h in enumerate(pbar):
#             supensemble = {'models': [], 'reps': []}
#             results = []
#             for train_idx, valid_idx in train_folds:
#                 ensemble = {'models': [], 'reps': []}

#                 for h_m in h:
#                     h_m = copy.deepcopy(h_m)
#                     arch = self.models[h_m['name']]
#                     train_x, train_y = x[h_m['representation']][train_idx], y[train_idx]
#                     ensemble['reps'].append(h_m['representation'])
#                     del h_m['name'],  h_m['representation']
#                     arch = arch(**h_m)
#                     arch.fit(train_x, train_y)
#                     ensemble['models'].append(arch)

#                 valid_y = y[valid_idx]
#                 preds = np.zeros(valid_y.shape)
#                 for arch, rep in zip(ensemble['models'], ensemble['reps']):
#                     valid_x = x[rep][valid_idx]
#                     preds += (arch.predict_proba(valid_x)[:, 1] /
#                               len(ensemble['models']))

#                 result = evaluate(preds, valid_y, self.optim_strategy['task'])
#                 results.append(result)
#                 supensemble['models'].extend(ensemble['models'])
#                 supensemble['reps'].extend(ensemble['reps'])

#             result_df = pd.DataFrame(results)
#             perf = result_df[self.metric].mean()
#             if ((self.optim_strategy['direction'] == 'minimize' and
#                 perf < self.best_metric) or
#                 (self.optim_strategy['direction'] == 'maximize' and
#                perf > self.best_metric)):
#                 self.best_metric = perf
#                 self.best_config = hpspace
#                 self.best_model = supensemble
#                 pbar.set_description(f'Best Value: {perf:.4g} at step {idx}')


class NoHpoTrainer(BaseTrainer):
    """
    Class `NoHpoTrainer` provides a training framework without hyperparameter optimization (HPO). 
    It assumes that the user provides a fixed set of hyperparameter configurations (`hpspace`) 
    for training and directly trains models using these configurations.

    Attributes:
        :type hpspace: List[dict]
        :param hpspace: A list of dictionaries, each specifying a model name, representation, 
                        and fixed hyperparameter configurations.

        :type optim_strategy: Dict[str, Any]
        :param optim_strategy: A dictionary specifying optimization-related properties, such as 
                               the task type (e.g., classification or regression).

    Example Schema for `hpspace`:
        ```python
        hpspace = [
            {
                'name': 'svm',
                'representation': 'representation1',
                'variables': {
                    'C': 1.0,
                    'kernel': 'linear',
                    'probability': True
                }
            },
            {
                'name': 'xgboost',
                'representation': 'representation2',
                'variables': {
                    'n_estimators': 100,
                    'learning_rate': 0.05,
                    'max_depth': 6
                }
            }
        ]
        ```

    Example Usage:
        ```python
        trainer = NoHpoTrainer(hpspace=my_hpspace, optim_strategy=my_strategy)
        trained_models = trainer.train(train_folds, x, y)
        ```
    """
    def __init__(self, hpspace: List[dict],
                 optim_strategy: Dict[str, Any], **args):
        """
        Initializes the trainer with the provided hyperparameter space and optimization strategy.

        :type hpspace: List[dict]
            :param hpspace: A list of dictionaries defining the models and their fixed hyperparameters.

        :type optim_strategy: Dict[str, Any]
            :param optim_strategy: Dictionary defining the task type and other optimization properties.
        """
        self.hpspace = hpspace
        self.optim_strategy = optim_strategy
        self.properties = copy.deepcopy(self.__dict__)
        self.models = self._import_models(
            optim_strategy['task'],
            [x['name'] for x in hpspace]
        )

    def train(
        self,
        train_folds: List[Tuple[np.ndarray, np.ndarray]],
        x: Dict[str, np.ndarray],
        y: np.ndarray
    ) -> Union[Callable, List[Callable]]:
        """
        Trains models using the provided fixed hyperparameter configurations and cross-validation folds.

        :type train_folds: List[Tuple[np.ndarray, np.ndarray]]
            :param train_folds: A list of training folds for cross-validation.

        :type x: Dict[str, np.ndarray]
            :param x: A dictionary mapping representations to feature matrices.

        :type y: np.ndarray
            :param y: The target labels for the dataset.

        :rtype: Union[Callable, List[Callable]]
            :return: A dictionary containing the trained models and their associated representations.
        """
        pbar = self.hpspace
        for idx, h_m in enumerate(pbar):
            supensemble = {'models': [], 'reps': []}
            results = []
            for train_idx, valid_idx in train_folds:
                ensemble = {'models': [], 'reps': []}
                arch = self.models[h_m['name']]
                train_x, train_y = x[h_m['representation']][train_idx], y[train_idx]
                ensemble['reps'].append(h_m['representation'])
                arch = arch(**h_m['variables'])
                arch.fit(train_x, train_y)
                ensemble['models'].append(arch)
                supensemble['models'].extend(ensemble['models'])
                supensemble['reps'].extend(ensemble['reps'])
        return supensemble
