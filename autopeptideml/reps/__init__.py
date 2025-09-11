from .engine import RepEngineBase
from .seq_based import RepEngineOnehot

PLMs = ['prot-t5-xxl', 'prot-t5-xl', 'protbert', 'prost-t5',
        'esm2-15b', 'esm2-3b', 'esm2-650m', 'esm1b', 'esm2-150m',
        'esm2-35m', 'esm2-8m', 'esmc-600m', 'esmc-300m', 'ankh-base',
        'ankh-large']
CLMs = ['molformer-xl', 'chemberta-2', 'peptideclm']
FPs = ['ecfp', 'morgan', 'fcfp', 'pepfunn']
