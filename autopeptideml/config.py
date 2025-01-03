import yaml
from typing import *

from .pipeline import Pipeline, CanonicalCleaner, CanonicalFilter
from .reps import RepEngineBase


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


def load_representation(rep_config: str) -> RepEngineBase:
    if 'LM' in rep_config['engine']:
        from .reps.lms import RepEngineLM
        re = RepEngineLM(rep_config['model'], rep_config['average_pooling'])
        re.batch_size = rep_config['batch_size']
    elif 'FP' in rep_config['engine']:
        from .reps.fps import RepEngineFP
        re = RepEngineFP(rep_config['fp'], nbits=rep_config['nbits'],
                         radius=rep_config['radius'])
    elif 'OneHot' in rep_config['engine']:
        from .reps import RepEngineOnehot
        re = RepEngineOnehot(rep_config['max_length'])
    return re


def load_modules(path: str):
    config = yaml.safe_load(open(path))
    pipe_config = config['pipeline']
    rep_config = config['representation']
    pipeline = load_pipeline(pipe_config)
    pipeline.save('trial.yml')
    rep = load_representation(rep_config)
    rep.save('trial_rep.yml')


if __name__ == "__main__":
    path = '/Users/raulfd/Projects/Finished/APML_Project/AutoPeptideML/autopeptideml/data/new_config.yaml' 
    load_modules(path)
