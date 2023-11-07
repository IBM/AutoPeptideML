from lightgbm import LGBMClassifier
import sklearn.ensemble
import sklearn.neighbors
import sklearn.neural_network
import sklearn.svm
import sklearn.metrics

from ..utils.unidl4biopep.model import Cnn


SUPPORTED_MODELS = {
    'knn': sklearn.neighbors.KNeighborsClassifier,
    'rfc': sklearn.ensemble.RandomForestClassifier,
    'svm': sklearn.svm.SVC,
    'mlp': sklearn.neural_network.MLPClassifier,
    'xgboost': sklearn.ensemble.GradientBoostingClassifier,
    'adaboost': sklearn.ensemble.AdaBoostClassifier,
    'unidl4biopep': Cnn,
    'lightgbm': LGBMClassifier
}

SYNONYMS = {
    'knn': ['knn', 'k-nearest neighbours', 'k-nearest neighbours'],
    'rfc': ['rf', 'rfc', 'random forest classifier', 'random forest'],
    'svm': ['svm', 'svc', 'support vector machine'],
    'mlp': ['mlp', 'multi-layer perceptron', 'ann', 'artificial neural network'],
    'xgboost': ['xgboost', 'xgboost classifier', 'extreme gradient boosting'],
    'adaboost': ['adaboost', 'adaptative gradient boosting'],
    'unidl4biopep': ['unidl4biopep'],
    'lightgbm': ['lightgbm', 'lightgbm classifier', 'light gradient boosted machine']
}
