import sklearn.metrics


METRICS = [
    'accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'f1_weighted',
    'jaccard', 'jaccard_weighted', 'matthews_corrcoef', 'precision',
    'precision_weighted', 'recall', 'recall_weighted', 'roc_auc'
]

METRIC2FUNCTION = {
    'accuracy': sklearn.metrics.accuracy_score,
    'balanced_accuracy': sklearn.metrics.balanced_accuracy_score,
    'f1': sklearn.metrics.f1_score,
    'f1_weighted': sklearn.metrics.f1_score,
    'precision': sklearn.metrics.precision_score,
    'precision_weighted': sklearn.metrics.precision_score,
    'recall': sklearn.metrics.recall_score,
    'roc_auc': sklearn.metrics.roc_auc_score,
    'average_precision': sklearn.metrics.average_precision_score,
    'jaccard': sklearn.metrics.jaccard_score,
    'jaccard_weighted': sklearn.metrics.jaccard_score,
    'matthews_corrcoef': sklearn.metrics.matthews_corrcoef,
    'recall_weighted': sklearn.metrics.recall_score
}

THRESHOLDED_METRICS = [
    'accuracy', 'balanced_accuracy', 'f1', 'f1_weighted',
    'jaccard', 'jaccard_weighted', 'matthews_corrcoef',
    'precision', 'precision_weighted', 'recall', 'recall_weighted'
]
