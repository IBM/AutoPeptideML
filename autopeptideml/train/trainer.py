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
    Implements hyperparameter optimization using Optuna for a variety of ML models and tasks.

    Inherits from `BaseTrainer` and supports both fixed and ensemble hyperparameter search spaces.
    The trainer searches for the best configuration using cross-validation or train/validation split.

    :param task: The machine learning task. Must be one of {"class", "reg", "multiclass"}.
    :type task: str
    :param direction: The direction of optimization. Either "maximize" or "minimize".
    :type direction: str
    :param metric: The evaluation metric. If None, defaults based on task.
    :type metric: Optional[str]
    :param ensemble: Whether to use ensemble modeling during optimization.
    :type ensemble: bool

    :ivar name: Name of the trainer (default: "optuna").
    :vartype name: str
    :ivar best_model: The best ensemble or model found during HPO.
    :vartype best_model: Optional[Callable]
    :ivar best_metric: Best performance achieved during optimization.
    :vartype best_metric: float
    :ivar best_config: The hyperparameter configuration corresponding to best_model.
    :vartype best_config: dict
    :ivar history: DataFrame with full optimization history.
    :vartype history: pd.DataFrame
    :ivar study: Optuna study object.
    :vartype study: optuna.study.Study
    """
    name = 'optuna'

    def _get_hpspace(self, models: List[str], custom_hpspace: dict) -> dict:
        """
        Constructs the full hyperparameter search space dictionary by combining
        custom and default model configurations.

        :param models: List of model names for which to build the hyperparameter space.
            If None, uses all available models.
            Typical values: ['svm', 'xgboost', 'lightgbm', 'catboost', 'cnn']
        :type models: List[str]

        :param custom_hpspace: Dictionary containing user-defined hyperparameter search spaces.
        :type custom_hpspace: dict

        :return: A dictionary representing the structured hyperparameter space.
        :rtype: dict
        """
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
        """
        Prepares the hyperparameter space for a single Optuna trial.

        Selects model(s), representations, and hyperparameter values based on the trial object.
        Supports both fixed and ensemble model types with conditional parameters.

        :param trial: An Optuna trial object from which hyperparameters are suggested.
        :type trial: optuna.trial.Trial

        :return: List of selected models with their hyperparameters and representation.
        :rtype: dict

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
        """
        Executes one hyperparameter optimization step using a configuration proposed by Optuna.

        Evaluates the selected model(s) across folds, tracks results, and updates best model if applicable.

        :param trial: The Optuna trial suggesting the hyperparameter configuration.
        :type trial: optuna.trial.Trial

        :return: Evaluation score computed from the selected metric (e.g., 'mcc', 'f1_weighted', 'mse').
        :rtype: float

        :raises ValueError: If the hyperparameter space is malformed or missing required fields.
        :raises KeyError: If conditional dependencies fail during configuration evaluation.
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
            self.best_run = result_df['run'].max()
            self.best_model = supensemble
        return perf

    def hpo(
        self,
        x: Union[Dict[str, np.ndarray], np.ndarray],
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
        """
        Runs Optuna-based hyperparameter optimization using the provided data and model list.

        :param x: Dictionary of representation names to feature arrays (shape: [n_samples, n_features]).
            It could also be an array.
        :type x: Union[Dict[str, np.ndarray], np.ndarray]
        :param y: Target values.
        :type y: np.ndarray
        :param models: List of model names to optimize. Defaults to ALL_MODELS.
        :type models: List[str]
        :param n_folds: Number of cross-validation folds. Ignored if `train_val_ratio` is specified.
        :type n_folds: int
        :param train_val_ratio: Proportion of training data to use for validation split (e.g., 0.2).
        :type train_val_ratio: float
        :param n_trials: Maximum number of Optuna trials to run.
        :type n_trials: int
        :param patience: Number of trials with no improvement before early stopping. Defaults to n_trials // 5.
        :type patience: int
        :param random_state: Seed for reproducibility.
        :type random_state: int
        :param n_jobs: Number of parallel jobs (used in model training if supported).
        :type n_jobs: int
        :param verbose: Verbosity level (0: silent, 1: warnings only, 2: show progress bar).
        :type verbose: int
        :param custom_hpspace: Custom hyperparameter space for specific models.
        :type custom_hpspace: dict
        :param db_file: Path to Optuna SQLite DB file for persistent study storage.
        :type db_file: Optional[str]
        :param study_name: Name for the Optuna study.
        :type study_name: Optional[str]

        :return: The best model or ensemble model found during HPO.
        :rtype: Union[Callable, List[Callable]]

        :raises ImportError: If Optuna is not installed.
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("This function requires optuna",
                              "Please try: `pip install optuna`")
        if verbose < 1:
            optuna.logging.set_verbosity(optuna.logging.ERROR)
        elif verbose < 3:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        if isinstance(x, np.ndarray):
            x = {'default': x}

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
            self.patience = n_trials // 5
        else:
            self.patience = patience
        self.best_metric = (float("inf") if self.direction == 'minimize'
                            else float('-inf'))

        # Data preparation
        self.train_folds = self._define_folds(x[list(x.keys())[0]], y,
                                              n_folds, train_val_ratio)
        self.x, self.y = x, y

        # Optuna definition
        if db_file is None:
            self.study = optuna.create_study(
                direction=self.direction,
                study_name=study_name if study_name is not None else None
            )
        else:
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
        """
        Loads an existing Optuna study from a SQLite database file.

        :param path: Path to the SQLite database file.
        :type path: str
        :param task: Task type used to initialize the trainer ("class", "reg", or "multiclass").
        :type task: str
        :param study_name: Name of the saved Optuna study.
        :type study_name: str

        :return: An OptunaTrainer instance with the loaded Optuna study.
        :rtype: OptunaTrainer

        :raises ImportError: If Optuna is not installed.
        """
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
