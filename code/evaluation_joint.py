import json
# import math
import operator
import os
import os.path as osp
from typing import Callable, Dict

import lightgbm
import numpy as np
import optuna
import pandas as pd
import sklearn.decomposition as decomp
import sklearn.ensemble as ensemble
import sklearn.gaussian_process as gauss
import sklearn.neighbors as knn
import sklearn.svm as svm
import typer
import xgboost
import yaml
import warnings

from hestia import HestiaGenerator, SimArguments
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (matthews_corrcoef, root_mean_squared_error,
                             accuracy_score, f1_score,
                             precision_score, recall_score, mean_squared_error,
                             mean_absolute_error)
# from represent_peptides import calculate_molformer
from tqdm import tqdm
# from autopeptideml.train.deep_learning.model import Cnn, MLP


def _pcc(preds, truths):
    return pearsonr(preds, truths)[0]


def _spcc(preds, truths):
    return spearmanr(preds, truths)[0]


def _f1_weighted(preds, truths):
    return f1_score(preds, truths, average='weighted')


def _recall(preds, truths):
    return recall_score(preds, truths, zero_division=True)


CLASSIFICATION_METRICS = {
    'mcc': matthews_corrcoef,
    'acc': accuracy_score,
    'f1': f1_score,
    'f1_weighted': _f1_weighted,
    'precision': precision_score,
    'recall': _recall
}

REGRESSION_METRICS = {
    'rmse': root_mean_squared_error,
    'mse': mean_squared_error,
    'mae': mean_absolute_error,
    'pcc': _pcc,
    'spcc': _spcc
}

CLASSIFICATION_MODELS = {
    'svm': svm.SVC,
    'knn': knn.KNeighborsClassifier,
    'gauss': gauss.GaussianProcessClassifier,
    'rf': ensemble.RandomForestClassifier,
    'xgboost': xgboost.XGBClassifier,
    'lightgbm': lightgbm.LGBMClassifier,
    # 'cnn': Cnn,
    # 'mlp': MLP
}

REGRESSION_MODELS = {
    'svm': svm.SVR,
    'knn': knn.KNeighborsRegressor,
    'gauss': gauss.GaussianProcessRegressor,
    'rf': ensemble.RandomForestRegressor,
    'xgboost': xgboost.XGBRegressor,
    'lightgbm': lightgbm.LGBMRegressor,
    # 'cnn': Cnn,
    # 'mlp': MLP
}


REGRESSION_TASKS = ['binding']
CLASSIFICATION_TASKS = ['cpp', 'antibacterial', 'antiviral']
TASKS = CLASSIFICATION_TASKS + REGRESSION_TASKS


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

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Do early stopping."""
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            study.stop()


def evaluate(preds, truth, pred_task) -> Dict[str, float]:
    result = {}
    if pred_task == 'reg':
        metrics = REGRESSION_METRICS
    else:
        metrics = CLASSIFICATION_METRICS

    for key, value in metrics.items():
        result[key] = value(preds, truth)
    return result


def define_hpspace(model: str, pred_task: str,
                   trial: optuna.Trial) -> dict:
    cwd = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(cwd, 'h_param_search',
                        f'{model}_{pred_task}.yml')

    hpspace = {}
    config = yaml.load(open(path), yaml.Loader)

    for key, item in config.items():
        if 'fixed' in item['type']:
            if item['type'].split('-')[1] == 'float':
                hpspace[key] = float(item['value'])
            elif item['type'].split('-')[1] == 'int':
                hpspace[key] = int(item['value'])
            else:
                hpspace[key] = item['value']
        elif item['type'] == 'bool':
            hpspace[key] = trial.suggest_categorical(
                key, choices=[True, False]
            )
        elif item['type'] == 'float':
            hpspace[key] = trial.suggest_float(
                key, low=float(item['min']),
                high=float(item['max']),
                log=bool(item['log'])
            )
        elif item['type'] == 'int':
            hpspace[key] = trial.suggest_int(
                key, low=int(item['min']),
                high=int(item['max']),
                log=bool(item['log'])
            )
        elif item['type'] == 'categorical':
            hpspace[key] = trial.suggest_categorical(
                key, choices=item['values']
            )
            if key == 'kernel':
                for name, value in item['extra_parameters'].items():
                    if name == hpspace[key]:
                        for subkey, subitem in value.items():
                            if subitem['type'] == 'fixed':
                                if subitem['type'].split('-')[1] == 'float':
                                    hpspace[subkey] = float(subitem['value'])
                                elif subitem['type'].split('-')[1] == 'int':
                                    hpspace[subkey] = int(subitem['value'])
                                else:
                                    hpspace[subkey] = subitem['value']
                            elif subitem['type'] == 'float':
                                hpspace[subkey] = trial.suggest_float(
                                    subkey, low=float(subitem['min']),
                                    high=float(subitem['max']),
                                    log=bool(subitem['log'])
                                )
                            elif subitem['type'] == 'int':
                                hpspace[subkey] = trial.suggest_int(
                                    subkey, low=int(subitem['min']),
                                    high=int(subitem['max']),
                                    log=bool(subitem['log'])
                                )
                            elif subitem['type'] == 'categorical':
                                hpspace[subkey] = trial.suggest_categorical(
                                    subkey, choices=subitem['values']
                                )
                            else:
                                raise ValueError("Subitem type: " +
                                                 f"{subitem['type']} " +
                                                 "does not exit.")

        else:
            raise ValueError(f"Item type: {item['type']} does not exit.")
    return hpspace


def hpo(pred_task: str, learning_algorithm: Callable,
        model_name: str,
        study: optuna.Study, train_x: np.ndarray,
        train_y: np.ndarray, valid_x: np.ndarray,
        valid_y: np.ndarray, n_trials: int,
        seed: int) -> dict:
    global best_model
    if pred_task == 'class':
        best_model = {'result': {'mcc': float('-inf')}}
    else:
        best_model = {'result': {'mse': float('inf')}}

    def hpo_objective(trial: optuna.Trial) -> float:
        global best_model
        if model_name not in ['cnn', 'mlp']:
            hpspace = define_hpspace(model_name, pred_task,
                                     trial)
        else:
            hpspace = {
                "optimizer": {
                    "lr": trial.suggest_float('lr', 1e-6, 1e1, log=True),
                    "weight_decay": trial.suggest_float('wd', 1e-10, 1e-3, log=True),
                    "optim_algorithm": "adam"},
                "logger": "cnn_log",
                "labels": 1 if pred_task == 'reg' else 2,
                "task": pred_task
            }
        if (not (model_name == 'svm' and pred_task == 'reg') and
           model_name != 'knn'):
            hpspace['random_state'] = seed
        if model_name in ['mlp', 'cnn']:
            del hpspace['random_state']
        model = learning_algorithm(**hpspace)
        if model_name in ['mlp', 'cnn']:
            model.fit(train_x, train_y, valid_x, valid_y, 'mps')
            preds = model.predict(valid_x, 'mps')
        else:
            model.fit(train_x, train_y)
            preds = model.predict(valid_x)

        result = evaluate(preds, valid_y, pred_task)
        if pred_task == 'class':
            if result['mcc'] > best_model['result']['mcc']:
                best_model = {
                    'model': model,
                    'config': hpspace,
                    'result': result
                }
            return result['mcc']
        else:
            if result['mse'] < best_model['result']['mse']:
                best_model = {
                    'model': model,
                    'config': hpspace,
                    'result': result
                }
            return -result['mse']
    if model_name == 'cnn' or model_name == 'mlp':
        study.optimize(hpo_objective, 20,
                       show_progress_bar=True,
                       gc_after_trial=False,
                       n_jobs=1)
    else:
        study.optimize(hpo_objective, n_trials,
                       callbacks=[EarlyStoppingCallback(20, direction='maximize')],
                       show_progress_bar=True, gc_after_trial=False, n_jobs=10)
    return best_model


def experiment(dataset: str, model: str, representation: str,
               type: str,
               hdg_c: HestiaGenerator,  hdg_nc: HestiaGenerator,
               df_c: pd.DataFrame, df_nc: pd.DataFrame,
               pca_pre: float, pca_post: float, n_trials: int = 100,
               seed: int = 1):
    global best_model

    np.random.seed(seed)
    if dataset in REGRESSION_TASKS:
        pred_task = 'reg'
        learning_algorithm = REGRESSION_MODELS[model]
    elif dataset in CLASSIFICATION_TASKS:
        pred_task = 'class'
        learning_algorithm = CLASSIFICATION_MODELS[model]
    else:
        raise ValueError(
            f"Dataset: {dataset} not in tasks: {', '.join(TASKS)}")
    x_type = {}
    for s in ['c', 'nc']:
        x_type[s] = []
        for idx, rep in enumerate(representation.split(',')):
            t_x = json.load(open(osp.join('reps', f'{rep}_{s}-{dataset}.json')))
            t_x = np.array(t_x)
            if pca_pre > 0.1:
                pca_calc = decomp.PCA(pca_pre, svd_solver='full')
                t_x = pca_calc.fit_transform(t_x)
            if idx == 0:
                x = t_x
            else:
                x = np.concatenate([x, t_x], axis=1)
        x_type[s] = x

    y_c = df_c.labels.to_numpy()
    y_nc = df_nc.labels.to_numpy()
    results = []
    c_partitions = hdg_c.get_partitions(filter=0.185, return_dict=True)
    nc_partitions = hdg_nc.get_partitions(filter=0.185, return_dict=True)

    for th, c_indc in c_partitions.items():
        if th not in nc_partitions:
            continue
        nc_indc = nc_partitions[th]

        print("THRESHOLD:", th)
        if th != 'random':
            if (th * 100) % 10 != 0:
                continue

        c_train_idx = c_indc['train']
        c_valid_idx = c_indc['valid']
        c_test_idx = c_indc['test']
        nc_train_idx = nc_indc['train']
        nc_valid_idx = nc_indc['valid']
        nc_test_idx = nc_indc['test']

        c_train_x, c_train_y = x_type['c'][c_train_idx], y_c[c_train_idx]
        nc_train_x, nc_train_y = x_type['nc'][nc_train_idx], y_nc[nc_train_idx]
        c_valid_x, c_valid_y = x_type['c'][c_valid_idx], y_c[c_valid_idx]
        nc_valid_x, nc_valid_y = x_type['nc'][nc_valid_idx], y_nc[nc_valid_idx]

        if type == 'joint':
            train_x = np.concatenate([c_train_x, nc_train_x])
            train_y = np.concatenate([c_train_y, nc_train_y])
            valid_x = np.concatenate([c_valid_x, nc_valid_x])
            valid_y = np.concatenate([c_valid_y, nc_valid_y])
        elif type == 'canonical':
            train_x, valid_x = c_train_x, c_valid_x
            train_y, valid_y = c_train_y, c_valid_y
        elif type == 'non-canonical':
            train_x, valid_x = nc_train_x, nc_valid_x
            train_y, valid_y = nc_train_y, nc_valid_y
        else:
            print(f"Type does not exist: {type}")
        c_test_x, nc_test_x = x_type['c'][c_test_idx], x_type['nc'][nc_test_idx]
        c_test_y, nc_test_y = y_c[c_test_idx], y_nc[nc_test_idx]

        study = optuna.create_study(direction='maximize')
        best_model = hpo(pred_task, learning_algorithm, model, study,
                         train_x, train_y, valid_x, valid_y, n_trials,
                         seed)

        preds_c = best_model['model'].predict(c_test_x)
        preds_nc = best_model['model'].predict(nc_test_x)

        result_c = evaluate(preds_c, c_test_y, pred_task)
        result_c = {f"{k}_c": v for k, v in result_c.items()}
        result_nc = evaluate(preds_nc, nc_test_y, pred_task)
        result_nc = {f"{k}_nc": v for k, v in result_nc.items()}
        result = result_c
        result.update(result_nc)
        result.update({'threshold': th, 'seed': seed})
        results.append(result)
        if pred_task == 'reg':
            print("C", result['spcc_c'])
            print("NC", result['spcc_nc'])

        else:
            print("C", result['mcc_c'])
            print("NC", result['mcc_nc'])

    result_df = pd.DataFrame(results)
    return result_df


def main(dataset: str, model: str, representation: str,
         type: str, pca_pre: float, pca_post: float, n_trials: int = 200,
         n_seeds: int = 5):
    if dataset == 'binding':
        similarity_metric_c = 'jaccard'
        radius_c = 4
        fp_c = 'mapc'
        similarity_metric_nc = 'jaccard'
        radius_nc = 10
        fp_nc = 'mapc'
    elif dataset == 'cpp':
        similarity_metric_c = 'jaccard'
        radius_c = 4
        fp_c = 'mapc'
        similarity_metric_nc = 'jaccard'
        radius_nc = 6
        fp_nc = 'mapc'
    elif dataset == 'antibacterial':
        similarity_metric_c = 'jaccard'
        radius_c = 4
        fp_c = 'mapc'
        similarity_metric_nc = 'tanimoto'
        radius_nc = 6
        fp_nc = 'ecfp'
    elif dataset == 'antiviral':
        similarity_metric_c = 'mmseqs+prefilter'
        radius_c = 0
        fp_c = 'na'
        similarity_metric_nc = 'jaccard'
        radius_nc = 6
        fp_nc = 'mapc'

    part_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'partitions'
    )
    data_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'downstream_data'
    )
    part_path_c = os.path.join(
        part_dir, f"c-{dataset}_{similarity_metric_c}_{fp_c}_{radius_c}.gz"
    )
    part_path_nc = os.path.join(
        part_dir, f"nc-{dataset}_{similarity_metric_nc}_{fp_nc}_{radius_nc}.gz"
    )
    os.makedirs(part_dir, exist_ok=True)

    df_c = pd.read_csv(osp.join(data_path, f'c-{dataset}.csv'))
    df_nc = pd.read_csv(osp.join(data_path, f'nc-{dataset}.csv'))
    df_c['name'] = dataset
    df_nc['name'] = dataset

    if os.path.exists(part_path_c):
        hdg_c = HestiaGenerator(df_c)
        hdg_c.from_precalculated(part_path_c)
        hdg_nc = HestiaGenerator(df_nc)
        hdg_nc.from_precalculated(part_path_nc)

    warnings.filterwarnings('ignore')
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    results_dir = os.path.join(
        os.path.dirname(__file__), '..', '..',
        'Results', type
    )
    results_path = os.path.join(
        results_dir,
        f'{dataset}_{model}_pre_{pca_pre}_post_{pca_post}_{representation}.csv'
    )
    os.makedirs(results_dir, exist_ok=True)

    results_df = pd.DataFrame()
    for i in range(n_seeds):
        print("SEED:", i)
        result_df = experiment(
            dataset, model, representation,
            type,
            hdg_c, hdg_nc,
            df_c, df_nc,
            pca_pre, pca_post,
            n_trials, i
        )
        results_df = pd.concat([results_df, result_df])
    results_df.to_csv(results_path, index=False)


if __name__ == '__main__':
    typer.run(main)
