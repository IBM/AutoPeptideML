from typing import Callable, Dict, List, Tuple

import numpy as np


SKLEARN_MODELS = ['knn', 'svm', 'rf', 'adaboost', 'gradboost']
ALL_MODELS = SKLEARN_MODELS + ['lightgbm', 'xgboost']


class VotingEnsemble:
    def __init__(self, models: List[Callable]):
        self.models = models

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        :param x: _description_
        :type x: np.ndarray
        :return: Mean, standard deviation
        :rtype: [np.ndarray, np.ndarray]
        """
        out = []
        for model in self.models:
            t_pred = model.predict(x)
            out.append(t_pred)
        pred = np.stack(out)
        out_mean = np.mean(pred, axis=0)
        out_std = np.std(pred, axis=0)
        return out_mean, out_std

    def predict_proba(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        :param x: _description_
        :type x: np.ndarray
        :return: Mean, standard deviation
        :rtype: [np.ndarray, np.ndarray]
        """
        out = []
        for model in self.models:
            t_pred = model.predict_proba(x)
            out.append(t_pred)
        pred = np.stack(out)
        out_mean = np.mean(pred, axis=0)
        out_std = np.std(pred, axis=0)
        return out_mean, out_std


def load_sklearn_models(task: str) -> Dict[str, Callable]:
    try:
        import sklearn as sk
    except ImportError:
        raise ImportError("This function requires scikit-learn",
                          "Please try: `pip install scikit-learn`")

    from sklearn import (svm, ensemble, neighbors)
    if 'class' in task:
        arch = {
            'knn': neighbors.KNeighborsClassifier,
            'svm': svm.SVC,
            'rf': ensemble.RandomForestClassifier,
            'adaboost': ensemble.AdaBoostClassifier,
            'gradboost': ensemble.GradientBoostingClassifier,

        }
    elif 'reg' in task:
        arch = {
            'knn': neighbors.KNeighborsRegressor,
            'svm': svm.SVR,
            'rf': ensemble.RandomForestRegressor,
            'adaboost': ensemble.AdaBoostRegressor,
            'gradboost': ensemble.GradientBoostingRegressor
        }
    else:
        raise NotImplementedError(
            f"Task type: {task} not implemented."
        )
    return arch


def load_lightgbm(task: str) -> Dict[str, Callable]:
    try:
        import lightgbm
    except ImportError:
        raise ImportError("This function requires lightgbm",
                          "Please try: `pip install lightgbm`")
    if 'class' in task:
        arch = {'lightgbm': lightgbm.LGBMClassifier}
    elif 'reg' in task:
        arch = {'lightgbm': lightgbm.LGBMRegressor}
    else:
        raise NotImplementedError(
            f"Task type: {task} not implemented."
        )
    return arch


def load_xgboost(task: str) -> Dict[str, Callable]:
    try:
        import xgboost
    except ImportError:
        raise ImportError("This function requires lightgbm",
                          "Please try: `pip install lightgbm`")
    if 'class' in task:
        arch = {'xgboost': xgboost.XGBClassifier}
    elif 'reg' in task:
        arch = {'xgboost': xgboost.XGBRegressor}
    else:
        raise NotImplementedError(
            f"Task type: {task} not implemented."
        )
    return arch


def load_torch(task: str) -> Dict[str, Callable]:
    try:
        from .deep_learning import Cnn
    except ImportError:
        raise ImportError("This function requires torch",
                          "Please try: `pip install torch`")

    return {"cnn": Cnn}
