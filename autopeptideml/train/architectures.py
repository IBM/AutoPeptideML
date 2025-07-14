import logging

from os import makedirs, listdir
from os import path as osp
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import onnxmltools as onxt
import onnxruntime as rt

from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx


SKLEARN_MODELS = ['knn', 'svm', 'rf', 'gradboost']
ALL_MODELS = SKLEARN_MODELS + ['lightgbm', 'xgboost', 'catboost']


class OnnxModel:
    def __init__(self, path: str):
        self.session = rt.InferenceSession(
            path, providers=['CPUExecutionProvider']
        )

    def predict(self, x: np.ndarray):
        input_dict = {"float_input": x.astype(np.float32)}
        preds = self.session.run(None, input_dict)
        return preds[0]

    def predict_proba(self, x: np.ndarray):
        input_dict = {"float_input": x.astype(np.float32)}
        preds = self.session.run(None, input_dict)
        return np.array([i[1] for i in preds[1]])


class VotingEnsemble:
    models: List[Callable]
    reps: List[str]
    dims: Dict[str, int] = None

    def __init__(self, models: List[Callable], reps: List[str]):
        self.models = models
        self.reps = reps

    def _stack_results(self, out: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        pred = np.stack(out)
        out_mean = np.mean(pred, axis=0)
        out_std = np.std(pred, axis=0)
        return out_mean, out_std

    def predict(self, x: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        :param x: _description_
        :type x: np.ndarray
        :return: Mean, standard deviation
        :rtype: [np.ndarray, np.ndarray]
        """
        out = []
        self.dims = {rep: x[rep].shape[1] for rep in self.reps}

        if isinstance(x, dict):
            for rep, model in zip(self.reps, self.models):
                t_pred = model.predict(x[rep])
                out.append(t_pred)
        else:
            for model in self.models:
                t_pred = model.predict(x)
                out.append(t_pred)
        return self._stack_results(out)

    def predict_proba(self, x: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        :param x: _description_
        :type x: np.ndarray
        :return: Mean, standard deviation
        :rtype: [np.ndarray, np.ndarray]
        """
        out = []
        self.dims = {rep: x[rep].shape[1] for rep in self.reps}

        if isinstance(x, dict):
            for rep, model in zip(self.reps, self.models):
                t_pred = model.predict_proba(x[rep])
                out.append(t_pred)
        else:
            for model in self.models:
                t_pred = model.predict_proba(x)
                out.append(t_pred)
        return self._stack_results(out)

    def save(self, path: str) -> None:
        if self.dims is None:
            raise RuntimeError("The `predict` or `predict_proba` method has to be called before saving the model.")

        if osp.isfile(path):
            raise FileExistsError(f"Path: {path} is a file.")
        else:
            makedirs(path, exist_ok=True)

        logger = logging.getLogger("skl2onnx")
        logger.setLevel(logging.DEBUG)
        for idx, (mdl, rep) in enumerate(zip(self.models, self.reps)):
            variable_type = FloatTensorType([None, self.dims[rep]])
            initial_types = [('float_input', variable_type)]
            if 'LGBM' in str(mdl):
                mdl_onnx = onxt.convert_lightgbm(
                    mdl, initial_types=initial_types
                )
            elif 'XGB' in str(mdl):
                mdl_onnx = onxt.convert_xgboost(
                    mdl, initial_types=initial_types
                )
            elif 'Cat' in str(mdl):
                mdl_onnx = onxt.convert_catboost(
                    mdl, initial_types=initial_types
                )
            else:
                mdl_onnx = to_onnx(mdl, initial_types=initial_types)

            name = f'{idx}_{rep}.onnx'
            with open(osp.join(path, name), 'wb') as f:
                f.write(mdl_onnx.SerializeToString())

    @classmethod
    def load(self, path) -> "VotingEnsemble":
        if not osp.isdir(path):
            raise NotADirectoryError(f"Path: {path} is not a directory.")
        models, reps = [], []
        for filepath in listdir(path):
            filepath = osp.join(path, filepath)
            if not filepath.endswith('.onnx'):
                raise RuntimeError(f"File: {filepath} in path: {path} is not ONNX model.")
            models.append(OnnxModel(filepath))
            reps.append(osp.basename(filepath).split("_")[1].split('.')[0])
        return VotingEnsemble(models, reps)


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


def load_catboost(task: str) -> Dict[str, Callable]:
    try:
        import catboost
    except ImportError:
        raise ImportError("This function requires catboost",
                          "Please try: `pip install catboost`")
    if 'class' in task:
        arch = {'catboost': catboost.CatBoostClassifier}
    elif 'reg' in task:
        arch = {'catboost': catboost.CatBoostRegressor}
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
