from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.svm
import sklearn.metrics
from xgboost import XGBClassifier

from ..utils.unidl4biopep.model import Cnn


SUPPORTED_MODELS = {
    'knn': KNeighborsClassifier,
    'rfc': RandomForestClassifier,
    'svm': sklearn.svm.SVC,
    'mlp': MLPClassifier,
    'xgboost': XGBClassifier,
    'adaboost': AdaBoostClassifier,
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
