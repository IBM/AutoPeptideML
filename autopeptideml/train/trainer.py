import copy
import itertools as it
import json
import operator
import warnings

import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import *
from .architectures import *
from .metrics import evaluate, CLASSIFICATION_METRICS, REGRESSION_METRICS


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


def choose_hps(variable, trial, hspace, model, key):
    if variable['type'] == 'int':
        hspace[key] = trial.suggest_int(
            f"{model}_{key}",
            variable['min'],
            variable['max'],
            log=variable['log']
        )

    elif variable['type'] == 'float':
        hspace[key] = trial.suggest_float(
            f"{model}_{key}",
            variable['min'],
            variable['max'],
            log=variable['log']
        )

    elif variable['type'] == 'categorical':
        hspace[key] = trial.suggest_categorical(
            f"{model}_{key}",
            variable['values']
        )

    elif variable['type'] == 'fixed':
        hspace[key] = variable['value']

    return hspace


class BaseTrainer:
    """
    Class `BaseTrainer` provides a framework for hyperparameter optimization (HPO) and model training. 
    It initializes models based on the specified task and optimization strategy, and serves as a base class for trainers.

    Attributes:
        :type name: str
        :param name: The name of the trainer. Set dynamically in derived classes.

        :type hpspace: Dict[str, Any]
        :param hpspace: The hyperparameter search space for the models.

        :type optim_strategy: Dict[str, Any]
        :param optim_strategy: The optimization strategy, including the task type and other configurations.

        :type best_config: dict
        :param best_config: The best configuration identified during HPO. Default is `None`.

        :type best_model: List[Dict[str, Union[str, Callable]]]
        :param best_model: The best-performing model(s) identified during training.
    """
    name: str
    hpspace: Dict[str, Any]
    optim_strategy: Dict[str, Any]
    best_config: dict = None
    best_model: List[Dict[str, Union[str, Callable]]]

    def __init__(self, hpspace: Dict[str, Any],
                 optim_strategy: Dict[str, Any], **args):
        """
        Initializes the trainer with the specified hyperparameter space and optimization strategy.

        :param hpspace: Dictionary describing the hyperparameter search space for the models.
        :type hpspace: Dict[str, Any]
        :param optim_strategy: Dictionary describing the optimization strategy,
            including the task type and other configurations.
        :type optim_strategy: Dict[str, Any]
        """
        self.hpspace = hpspace
        self.optim_strategy = optim_strategy
        self.properties = copy.deepcopy(self.__dict__)
        self.models = self._import_models(
            optim_strategy['task'],
            hpspace['models']['elements'].keys()
        )

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
            'representations': ['representation1', 'representation2']
        }
        ```
    """
    name = 'optuna'

    def _prepare_hpspace(self, trial) -> dict:
        """Prepares the hyperparameter space for a given Optuna trial.

        :type trial: optuna.trial.Trial
            :param trial: An Optuna trial object used to suggest hyperparameter values.

        :rtype: dict
            :return: A dictionary containing the hyperparameter configurations for the models.

        :raises KeyError: If the hyperparameter space is not properly defined.
        """
        full_hspace = []
        if (self.hpspace['models']['type'] == 'fixed' or
           self.hpspace['models']['type'] == 'ensemble'):
            for m_key, model in self.hpspace['models']['elements'].items():
                hspace = {}
                for key, variable in model.items():
                    if 'condition' in variable:
                        continue

                    hspace = choose_hps(variable, trial, hspace, model,
                                        key)
                for key, variable in model.items():
                    if 'condition' not in variable:
                        continue

                    hspace = choose_hps(variable, trial, hspace, model,
                                        key)
                    if 'condition' in variable:
                        conditions = variable['condition'].split('|')
                        for condition in conditions:
                            condition = condition.split('-')
                            v, f = condition[0], condition[1]
                            if hspace[v] != f:
                                del hspace[v]
                                break

                full_hspace.append(
                    {'name': m_key, 'variables': hspace,
                     'representation': trial.suggest_categorical(
                        f"{m_key}_rep", self.hpspace['representations'])})

        else:
            models = []
            for m_key, model in self.hpspace['models']['elements'].items():
                models.append(m_key)
            model = trial.suggest_categorical('model', models)
            hspace = {}

            for key, variable in self.hpspace['models']['elements'][model].items():
                if 'condition' in variable:
                    continue
                hspace = choose_hps(variable, trial, hspace, model, key)

            for key, variable in self.hpspace['models']['elements'][model].items():
                if 'condition' not in variable:
                    continue
                hspace = choose_hps(variable, trial, hspace, model, key)

                conditions = variable['condition'].split('|')
                for condition in conditions:
                    condition = condition.split('-')
                    v, f = condition[0], condition[1]
                    if hspace[v] != f:
                        del hspace[v]
                        break

            full_hspace.append(
                {'name': model, 'variables': hspace,
                 'representation': trial.suggest_categorical(
                     f"{model}_rep", self.hpspace['representations']
                 )})
        return full_hspace

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
            hspace = self._prepare_hpspace(trial)
        except KeyError:
            raise ValueError(
                "Hyperparameter space is not properly defined.",
                "Please check the definition of all fields."
            )
        warnings.filterwarnings('ignore')
        results, supensemble = [], {'models': [], 'reps': []}
        train_folds = self.train_folds
        x = self.x
        y = self.y

        for train_idx, valid_idx in train_folds:
            ensemble = {'models': [], 'reps': []}

            for h_m in hspace:
                arch = self.models[h_m['name']]
                if self.optim_strategy['task'] == 'reg' and h_m['name'] == 'svm':
                    if 'probability' in h_m['variables']:
                        del h_m['variables']['probability']
                arch = arch(**h_m['variables'])
                train_x, train_y = x[h_m['representation']][train_idx], y[train_idx]
                with warnings.catch_warnings():
                    arch.fit(train_x, train_y)
                ensemble['models'].append(arch)
                ensemble['reps'].append(h_m['representation'])

            valid_y = y[valid_idx]
            preds = np.zeros(valid_y.shape)
            for arch, rep in zip(ensemble['models'], ensemble['reps']):
                valid_x = x[rep][valid_idx]
                try:
                    preds += (arch.predict_proba(valid_x)[:, 1] /
                              len(ensemble['models']))
                except AttributeError:
                    preds += (arch.predict(valid_x)[:] /
                              len(ensemble['models']))

            result = evaluate(preds, valid_y, self.optim_strategy['task'])
            results.append(result)
            supensemble['models'].extend(ensemble['models'])
            supensemble['reps'].extend(ensemble['reps'])

        result_df = pd.DataFrame(results)
        perf = result_df[self.metric].mean()
        if ((self.optim_strategy['direction'] == 'minimize' and
             perf < self.best_metric) or
           (self.optim_strategy['direction'] == 'maximize' and
           perf > self.best_metric)):

            self.best_metric = perf
            self.best_config = hspace
            self.best_model = supensemble
        return perf

    def hpo(
        self,
        train_folds: List[Tuple[np.ndarray, np.ndarray]],
        x: Dict[str, np.ndarray],
        y: np.ndarray
    ) -> Union[Callable, List[Callable]]:
        """
        Performs hyperparameter optimization using Optuna.

        :type train_folds: List[Tuple[np.ndarray, np.ndarray]]
            :param train_folds: A list of training folds for cross-validation.

        :type x: Dict[str, np.ndarray]
            :param x: A dictionary mapping representations to feature matrices.

        :type y: np.ndarray
            :param y: The target labels for the dataset.

        :rtype: Union[Callable, List[Callable]]
            :return: The best-performing model(s) after optimization.

        :raises ImportError: If Optuna is not installed.
        :raises ValueError: If required fields in the hyperparameter space are missing.
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("This function requires optuna",
                              "Please try: `pip install optuna`")
        # optuna.logging.set_verbosity(optuna.logging.ERROR)
        self.best_model = None
        self.best_metric = (
            float("inf") if self.optim_strategy['direction'] == 'minimize'
            else float('-inf')
        )
        study = optuna.create_study(
            direction=self.optim_strategy['direction']
        )
        self.metric = self.optim_strategy['metric']
        if 'patience' in self.optim_strategy:
            callback = EarlyStoppingCallback(
                early_stopping_rounds=self.optim_strategy['patience'],
                direction=self.optim_strategy['direction']
            )
        else:
            callback = EarlyStoppingCallback(
                early_stopping_rounds=self.optim_strategy['n_steps'],
                direction=self.optim_strategy['direction']
            )
        self.train_folds = train_folds
        self.x = x
        self.y = y
        study.optimize(self._hpo_step, n_trials=self.optim_strategy['n_steps'],
                       n_jobs=self.optim_strategy['n_jobs'],
                       callbacks=[callback], gc_after_trial=True,
                       show_progress_bar=True)


class GridTrainer(BaseTrainer):
    """
    Class `GridTrainer` implements a grid search-based hyperparameter optimization (HPO) framework.
    It systematically explores a predefined hyperparameter space by evaluating all possible combinations.

    Attributes:
        :type name: str
        :param name: The name of the trainer. Default is `'grid'`.

    Example Usage:
        ```python
        trainer = GridTrainer(hpspace=my_hpspace, optim_strategy=my_strategy)
        best_model = trainer.hpo(train_folds, x, y)
        ```

    Example Schema for `hpspace`:

        ```python
        hpspace = {
            'models': {
                'type': 'fixed',  # Options: 'fixed', 'ensemble'
                'elements': {
                    'svm': [
                        {'C': {'type': 'float', 'min': 0.1, 'max': 10, 'steps': 5, 'log': True}},
                        {'kernel': {'type': 'categorical', 'values': ['linear', 'rbf']}},
                        {'probability': {'type': 'fixed', 'value': True}}
                    ],
                    'xgboost': [
                        {'n_estimators': {'type': 'int', 'min': 50, 'max': 200, 'steps': 4}},
                        {'learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.1, 'steps': 5}},
                        {'max_depth': {'type': 'int', 'min': 3, 'max': 10, 'steps': 3}}
                    ]
                }
            },
            'representations': ['representation1', 'representation2']
        }
        ```
    """
    name = 'grid'

    def _prepare_hpspace(self) -> dict:
        """
        Prepares the hyperparameter space for grid search by generating all possible combinations
        of hyperparameter values.

        :rtype: dict
            :return: A list of dictionaries representing all possible hyperparameter configurations.
        """
        full_hspace = []
        for m_key, model in self.hpspace['models']['elements'].items():
            hspace = {}
            for key in model:
                variable = model[key]

                if variable['type'] == 'int':
                    min, max = variable['min'], variable['max']
                    if variable['log']:
                        min, max = np.log10(min), np.log10(max)
                        step = (max - min) / (variable['steps'] - 1)
                    hspace[key] = []
                    for i in range(variable['steps']):
                        val = min + step * i
                        if variable['log']:
                            hspace[key].append(int(10**val))
                        else:
                            hspace[key].append(int(val))

                elif variable['type'] == 'float':
                    min, max = variable['min'], variable['max']
                    if variable['log']:
                        min, max = np.log10(min), np.log10(max)
                        step = (max - min) / (variable['steps'] - 1)
                    hspace[key] = []
                    for i in range(variable['steps']):
                        val = min + step * i
                        if variable['log']:
                            hspace[key].append(10**val)
                        else:
                            hspace[key].append(val)

                elif variable['type'] == 'categorical':
                    hspace[key] = variable['values']

                elif variable['type'] == 'fixed':
                    hspace[key] = [variable['value']]

                if key.lower() in PROBABILITY:
                    hspace['probability'] = [True]
                if key.lower() in PROBABILITY and self.optim_strategy['task'] == 'reg':
                    del hspace['probability']

            full_hspace.append(
                {'name': m_key, 'variables': hspace,
                 'representation': self.hpspace['representations']})

        real_hspace = []
        for m_space in full_hspace:
            model = m_space['name']
            variables = [v for v in m_space['variables'].values()]
            names = list(m_space['variables'].keys())
            names.append('representation')
            combs = it.product(*variables, m_space['representation'])
            hspace = [{n: v[idx] for idx, n in enumerate(names)}
                      for v in combs]
            for idx in range(len(hspace)):
                hspace[idx]['name'] = model
            real_hspace.extend(hspace)
        return real_hspace

    def hpo(
        self,
        train_folds: List[Tuple[np.ndarray, np.ndarray]],
        x: Dict[str, np.ndarray],
        y: np.ndarray
    ) -> Union[Callable, List[Callable]]:
        """Performs hyperparameter optimization using grid search.

        :type train_folds: List[Tuple[np.ndarray, np.ndarray]]
            :param train_folds: A list of training folds for cross-validation.

        :type x: Dict[str, np.ndarray]
            :param x: A dictionary mapping representations to feature matrices.

        :type y: np.ndarray
            :param y: The target labels for the dataset.

        :rtype: Union[Callable, List[Callable]]
            :return: The best-performing model(s) after optimization.
        """
        self.best_model = None
        self.best_metric = (
            float("inf") if self.optim_strategy['direction'] == 'minimize'
            else float('-inf')
        )
        self.metric = self.optim_strategy['metric']
        hspace = self._prepare_hpspace()
        if (self.hpspace['models']['type'] == 'fixed' or
           self.hpspace['models']['type'] == 'ensemble'):
            o_hspace = copy.deepcopy(hspace)
            n_hspace = {}
            for e in o_hspace:
                if e['name'] in n_hspace:
                    n_hspace[e['name']].append(e)
                else:
                    n_hspace[e['name']] = [e]

            hspace = list(it.product(*list(n_hspace.values())))

        pbar = tqdm(hspace)
        for idx, h in enumerate(pbar):
            supensemble = {'models': [], 'reps': []}
            results = []
            for train_idx, valid_idx in train_folds:
                ensemble = {'models': [], 'reps': []}

                for h_m in h:
                    h_m = copy.deepcopy(h_m)
                    arch = self.models[h_m['name']]
                    train_x, train_y = x[h_m['representation']][train_idx], y[train_idx]
                    ensemble['reps'].append(h_m['representation'])
                    del h_m['name'],  h_m['representation']
                    arch = arch(**h_m)
                    arch.fit(train_x, train_y)
                    ensemble['models'].append(arch)

                valid_y = y[valid_idx]
                preds = np.zeros(valid_y.shape)
                for arch, rep in zip(ensemble['models'], ensemble['reps']):
                    valid_x = x[rep][valid_idx]
                    preds += (arch.predict_proba(valid_x)[:, 1] /
                              len(ensemble['models']))

                result = evaluate(preds, valid_y, self.optim_strategy['task'])
                results.append(result)
                supensemble['models'].extend(ensemble['models'])
                supensemble['reps'].extend(ensemble['reps'])

            result_df = pd.DataFrame(results)
            perf = result_df[self.metric].mean()
            if ((self.optim_strategy['direction'] == 'minimize' and
                perf < self.best_metric) or
                (self.optim_strategy['direction'] == 'maximize' and
               perf > self.best_metric)):
                self.best_metric = perf
                self.best_config = hspace
                self.best_model = supensemble
                pbar.set_description(f'Best Value: {perf:.4g} at step {idx}')


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
