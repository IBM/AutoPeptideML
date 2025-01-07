import copy
import json
import operator
import yaml
import warnings

import numpy as np
import pandas as pd

from typing import *
from .architectures import *
from .metrics import evaluate, CLASSIFICATION_METRICS, REGRESSION_METRICS


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


class BaseTrainer:
    name: str
    hpspace: Dict[str, Any]
    optim_strategy: Dict[str, Any]

    def __init__(self, hpspace: Dict[str, Any],
                 optim_strategy: Dict[str, Any], **args):

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

    def __str__(self) -> str:
        return json.dumps(self.properties)

    def hpo(
        self,
        train_folds: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Union[Callable, List[Callable]]:
        raise NotImplementedError


class OptunaTrainer(BaseTrainer):
    name = 'optuna'

    def _prepare_hpspace(self, trial) -> dict:
        NO_N_JOBS = ['svm', 'mlp', 'xgboost']
        PROBABILITY = ['svm']
        full_hspace = []
        if (self.hpspace['models']['type'] == 'fixed' or
           self.hpspace['models']['type'] == 'ensemble'):
            for m_key, model in self.hpspace['models']['elements'].items():
                hspace = {}
                for variable in model:
                    key = list(variable.keys())[0]
                    variable = variable[key]
                    if variable['type'] == 'int':
                        hspace[key] = trial.suggest_int(
                            f"{m_key}_{key}",
                            variable['min'],
                            variable['max'],
                            log=variable['log']
                        )

                    elif variable['type'] == 'float':
                        hspace[key] = trial.suggest_float(
                            f"{m_key}_{key}",
                            variable['min'],
                            variable['max'],
                            log=variable['log']
                        )

                    elif variable['type'] == 'categorical':
                        hspace[key] = trial.suggest_categorical(
                            f"{m_key}_{key}",
                            variable['values']
                        )

                    elif variable['type'] == 'fixed':
                        hspace[key] = variable['value']

                    if key.lower() in PROBABILITY:
                        hspace['probability'] = True

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

            for variable in self.hpspace['models']['elements'][model]:
                key = list(variable.keys())[0]
                variable = variable[key]
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

                if key.lower() in PROBABILITY:
                    hspace['probability'] = True

            full_hspace.append(
                {'name': model, 'variables': hspace,
                 'representation': trial.suggest_categorical(
                     f"{model}_rep", self.hpspace['representations']
                 )})

        return full_hspace

    def _hpo_step(self, trial) -> dict:
        try:
            hspace = self._prepare_hpspace(trial)
        except KeyError:
            raise ValueError(
                "Hyperparameter space is not properly defined.",
                "Please check the definition of all fields."
            )
        warnings.filterwarnings('ignore')
        results, supensemble = [], []
        train_folds = self.train_folds
        x = self.x
        y = self.y
        for train_idx, valid_idx in train_folds:
            train_x, train_y = x[train_idx], y[train_idx]
            valid_x, valid_y = x[valid_idx], y[valid_idx]
            ensemble = []
            for h_m in hspace:
                arch = self.models[h_m['name']]
                arch = arch(**h_m['variables'])
                arch.fit(train_x, train_y)
                ensemble.append(arch)

            preds = np.zeros(valid_y.shape)
            for arch in ensemble:
                preds += arch.predict_proba(valid_x)[:, 1] / len(ensemble)

            result = evaluate(preds, valid_y, self.optim_strategy['task'])
            results.append(result)
            supensemble.extend(ensemble)

        result_df = pd.DataFrame(results)
        perf = result_df[self.metric].mean()
        if ((self.optim_strategy['direction'] == 'minimize' and
             perf > self.best_metric) or
           (self.optim_strategy['direction'] == 'maximize' and
           perf < self.best_metric)):

            self.best_metric = perf
            self.best_model = supensemble
        return perf

    def hpo(
        self,
        train_folds: List[Tuple[np.ndarray, np.ndarray]],
        x: np.ndarray,
        y: np.ndarray
    ) -> Union[Callable, List[Callable]]:
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
