import yaml
from typing import *

import pandas as pd

from .pipeline import Pipeline, CanonicalCleaner, CanonicalFilter
from .reps import RepEngineBase
from .train import BaseTrainer, OptunaTrainer
from .autopeptideml import AutoPeptideML


def load_pipeline(pipe_config: dict) -> Pipeline:
    elements = []

    for config in pipe_config['elements']:
        name = list(config.keys())[0]
        if 'pipe' in name:
            item = load_pipeline(config[name])
        else:
            if 'filter-smiles' in config:
                from .pipeline.smiles import FilterSMILES
                config = config['filter-smiles']
                config = {} if config is None else config
                item = FilterSMILES(**config)

            elif 'sequence-to-smiles' in config:
                from .pipeline.smiles import SequenceToSMILES
                config = config['sequence-to-smiles']
                config = {} if config is None else config
                item = SequenceToSMILES(**config)

            elif 'canonical-cleaner' in config:
                config = config['canonical-cleaner']
                config = {} if config is None else config
                item = CanonicalCleaner(**config)

            elif 'canonical-filter' in config:
                config = config['canonical-filter']
                config = {} if config is None else config
                item = CanonicalFilter(**config)

        elements.append(item)
    return Pipeline(name=pipe_config['name'], elements=elements,
                    aggregate=pipe_config['aggregate'])


def load_representation(rep_config: str) -> Dict[str, RepEngineBase]:
    out = {}
    for r in rep_config['elements']:
        name = list(r.keys())[0]
        r_config = r[name]
        if 'LM' in r_config['engine']:
            from .reps.lms import RepEngineLM
            re = RepEngineLM(r_config['model'], r_config['average_pooling'])
            re.batch_size = r_config['batch_size']
            re.device = r_config['device']

        elif 'FP' in r_config['engine']:
            from .reps.fps import RepEngineFP
            re = RepEngineFP(r_config['fp'], nbits=r_config['nbits'],
                             radius=r_config['radius'])

        elif 'OneHot' in r_config['engine']:
            from .reps import RepEngineOnehot
            re = RepEngineOnehot(rep_config['max_length'])
        out[name] = re
    return out


def load_trainer(train_config: dict) -> BaseTrainer:
    hspace = train_config['hspace']
    optim_strategy = train_config['optim_strategy']
    if optim_strategy['trainer'] == 'optuna':
        trainer = OptunaTrainer(hspace, optim_strategy)
    return trainer


def load_modules(path: str):
    config = yaml.safe_load(open(path))
    pipe_config = config['pipeline']
    rep_config = config['representation']
    train_config = config['train']
    print(train_config)
    pipeline = load_pipeline(pipe_config)
    pipeline.save('trial.yml')
    rep = load_representation(rep_config)
    train = load_trainer(train_config)
    return pipeline, rep, train


if __name__ == "__main__":
    import os.path as osp
    import numpy as np

    path = '/Users/raulfd/Projects/Finished/APML_Project/AutoPeptideML/autopeptideml/data/new_config.yaml' 
    pipeline, reps, train = load_modules(path)

    df = pd.read_csv('./downstream_data/BBP.csv')
    apml = AutoPeptideML(verbose=False)
    datasets = apml.train_test_partition(
        df=df, test_size=0.2, threshold=0.5,
        denominator='longest', alignment='peptides'
    )
    folds = apml.train_val_partition(datasets['train'])
    folds = [(f['train'].index.to_numpy(),
              f['val'].index.to_numpy()) for f in folds]
    for name, rep in reps.items():
        if osp.exists(f'{rep.name}.pckl'):
            x = np.load(f'{rep.name}.pckl', allow_pickle=True)
        else:
            x = pipeline(df.sequence, n_jobs=10, verbose=False)
            if 'lm' in rep.name:
                rep.move_to_device(rep.device)
            x = rep.compute_reps(x, verbose=True, batch_size=rep.batch_size)
            x.dump(f'{rep.name}.pckl')

    train.hpo(folds, x, df.Y.to_numpy())
