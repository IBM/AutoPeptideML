import logging

from os import makedirs, listdir
from os import path as osp
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import onnxmltools as onxt
import onnxruntime as rt

from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx


SKLEARN_MODELS = ['knn', 'svm', 'rf', 'gradboost', 'logreg', 'linreg']
ALL_MODELS = SKLEARN_MODELS + ['lightgbm', 'xgboost']


class OnnxModel:
    """
    A wrapper around an ONNX model for inference using ONNX Runtime.

    Assumes that the ONNX model expects a single input named `"float_input"` of type float32.

    :param path: File path to the ONNX model.
    :type path: str
    """
    def __init__(self, path: str):
        so = rt.SessionOptions()
        so.log_severity_level = 3  # 0 = verbose, 1 = info, 2 = warning, 3 = error, 4 = fatal
        self.session = rt.InferenceSession(
            path, providers=['CPUExecutionProvider'],
            sess_options=so
        )

    def predict(self, x: np.ndarray):
        """
        Runs inference on input data and returns raw predictions (e.g., class labels or regression outputs).

        :param x: Input array of shape (n_samples, n_features).
                Must match the input shape expected by the ONNX model.
        :type x: np.ndarray
        :return: Predicted values from the ONNX model.
        :rtype: np.ndarray
        """
        try:
            input_dict = {"float_input": x.astype(np.float32)}
            preds = self.session.run(None, input_dict)
        except ValueError:
            input_dict = {"X": x.astype(np.float32)}
            preds = self.session.run(None, input_dict)
        return preds[0]

    def predict_proba(self, x: np.ndarray):
        """
        Runs inference on input data and returns class probabilities.

        This method assumes the ONNX model returns two outputs:
        - The first output (ignored here) is typically raw predictions or class indices.
        - The second output is a list of probability vectors, where each entry is a list of probabilities
        for each class. This method returns the probability for the positive class (index 1) only.

        :param x: Input array of shape (n_samples, n_features).
                Must match the input shape expected by the ONNX model.
        :type x: np.ndarray
        :return: Array of predicted probabilities for the positive class (shape: [n_samples]).
        :rtype: np.ndarray
        """
        try:
            input_dict = {"float_input": x.astype(np.float32)}
            preds = self.session.run(None, input_dict)
        except ValueError:
            input_dict = {"X": x.astype(np.float32)}
            preds = self.session.run(None, input_dict)
        return np.array([i[1] for i in preds[1]])


class VotingEnsemble:
    """
    A model ensemble that combines predictions from multiple sub-models trained on different feature representations.

    The ensemble averages predictions and provides both the mean and standard deviation across models.

    Supported model types for ONNX export include:

    - Scikit-learn: 'knn', 'svm', 'rf', 'gradboost'
    - Gradient Boosting: 'lightgbm', 'xgboost'

    Models must be compatible with skl2onnx, onnxmltools, or appropriate ONNX converters.

    Input format during prediction must match the reps provided at initialization.

    :param models: List of trained models, each conforming to a scikit-learn-style API (with `.predict` and `.predict_proba`).
    :type models: List[Callable]
    :param reps: List of string identifiers corresponding to the representation keys for each model.
    :type reps: List[str]
    """
    models: List[Callable]
    reps: List[str]
    dims: Dict[str, int] = None

    def __init__(self, models: List[Callable], reps: List[str]):
        self.models = models
        self.reps = reps

    def _stack_results(self, out: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes mean and standard deviation across model predictions.

        :param out: List of prediction arrays from individual models.
        :type out: List[np.ndarray]
        :return: Tuple of mean and standard deviation across models.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        pred = np.stack(out)
        out_mean = np.mean(pred, axis=0)
        out_std = np.std(pred, axis=0)
        return out_mean, out_std

    def predict(self, x: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts class labels or regression values using ensemble of models. Returns both mean and standard deviation.

        :param x: Input data. Can be:
                - A NumPy array if all models use the same representation.
                - A dictionary of representation name to NumPy array if models use different representations.
        :type x: Union[np.ndarray, Dict[str, np.ndarray]]
        :return: Tuple of (mean predictions, standard deviation across models).
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        out = []

        if isinstance(x, dict):
            self.dims = {rep: x[rep].shape[1] for rep in self.reps}
            for rep, model in zip(self.reps, self.models):
                t_pred = model.predict(x[rep])
                out.append(t_pred)
        else:
            self.dims = {rep: x.shape[1]}

            for model in self.models:
                t_pred = model.predict(x)
                out.append(t_pred)
        return self._stack_results(out)

    def predict_proba(self, x: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts class probabilities using ensemble of models. Returns both mean and standard deviation.

        :param x: Input data. Can be:
                - A NumPy array if all models use the same representation.
                - A dictionary of representation name to NumPy array if models use different representations.
        :type x: Union[np.ndarray, Dict[str, np.ndarray]]
        :return: Tuple of (mean probabilities, standard deviation across models).
        :rtype: Tuple[np.ndarray, np.ndarray]
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
        """
        Saves each model in the ensemble to ONNX format in the specified directory.

        The method infers input dimensions from prior calls to `predict` or `predict_proba`.

        :param path: Path to a directory where ONNX models will be saved. Must not point to a file.
        :type path: str
        :raises RuntimeError: If `predict` or `predict_proba` has not been called to determine input dimensions.
        :raises FileExistsError: If the given path is a file.
        """
        if self.dims is None:
            raise RuntimeError("The `predict` or `predict_proba` method has to be called before saving the model.")

        if osp.isfile(path):
            raise FileExistsError(f"Path: {path} is a file.")
        else:
            makedirs(path, exist_ok=True)

        logger = logging.getLogger("skl2onnx")
        logger.setLevel(logging.DEBUG)
        for idx, (mdl, rep) in enumerate(zip(self.models, self.reps)):
            variable_type = onxt.convert.common.data_types.FloatTensorType(
                [None, self.dims[rep]]
            )
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
                variable_type = FloatTensorType([None, self.dims[rep]])
                initial_types = [('float_input', variable_type)]
                mdl_onnx = to_onnx(mdl, initial_types=initial_types)

            name = f'{idx}_{rep}.onnx'
            with open(osp.join(path, name), 'wb') as f:
                f.write(mdl_onnx.SerializeToString())

    @classmethod
    def load(self, path) -> "VotingEnsemble":
        """
        Loads an ensemble from a directory of ONNX model files.

        Expects each file in the directory to be named as ``{index}_{rep}.onnx``,
        where ``rep`` is the representation used by that model.

        :param path: Path to a directory containing saved ONNX models.
        :type path: str
        :raises NotADirectoryError: If the path does not point to a valid directory.
        :raises RuntimeError: If any file in the directory is not an ONNX model.
        :return: A `VotingEnsemble` instance with models restored from ONNX.
        :rtype: VotingEnsemble
        """
        if not osp.isdir(path):
            raise NotADirectoryError(f"Path: {path} is not a directory.")
        models, reps = [], []
        for filepath in listdir(path):
            filepath = osp.join(path, filepath)
            if not filepath.endswith('.onnx'):
                raise RuntimeError(f"File: {filepath} in path: {path} is not ONNX model.")
            models.append(OnnxModel(filepath))
            reps.append(osp.basename(filepath).split("_")[1].split('.')[0])
        reps = [r if r != 'class' else 'esm2-8m' for r in reps]
        return VotingEnsemble(models, reps)


def load_sklearn_models(task: str) -> Dict[str, Callable]:
    try:
        import sklearn as sk
    except ImportError:
        raise ImportError("This function requires scikit-learn",
                          "Please try: `pip install scikit-learn`")

    from sklearn import (svm, ensemble, neighbors, linear_model)
    if 'class' in task:
        arch = {
            'knn': neighbors.KNeighborsClassifier,
            'svm': svm.SVC,
            'rf': ensemble.RandomForestClassifier,
            'gradboost': ensemble.GradientBoostingClassifier,
            'logreg': linear_model.LogisticRegression

        }
    elif 'reg' in task:
        arch = {
            'knn': neighbors.KNeighborsRegressor,
            'svm': svm.SVR,
            'rf': ensemble.RandomForestRegressor,
            'adaboost': ensemble.AdaBoostRegressor,
            'gradboost': ensemble.GradientBoostingRegressor,
            'linreg': linear_model.LinearRegression
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
