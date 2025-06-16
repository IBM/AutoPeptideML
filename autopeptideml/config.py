import yaml
import os
import os.path as osp

from multiprocessing import cpu_count

import pandas as pd

from ItsPrompt.prompt import Prompt


HP_SPACES = {
    "knn": {
        "n_neighbors": {
            "type": "int",
            "min": 1,
            "max": 20,
            "log": False
        },
        "weights": {
            "type": "categorical",
            "values": ["uniform", "distance"]
        }
    },
    "adaboost": {
        "n_estimators": {
            "type": "int",
            "min": 10,
            "max": 1000,
            "log": False
        },
        "learning_rate": {
            "type": "float",
            "min": 1e-7,
            "max": 1e-1,
            "log": True
        }
    },
    "gradboost": {
        "learning_rate": {
            "type": "float",
            "min": 1e-5,
            "max": 1e-1,
            "log": True
        },
        "n_estimators": {
            "type": "int",
            "min": 10,
            "max": 1000,
            "log": False
        },
        "min_samples_split": {
            "type": "int",
            "min": 2,
            "max": 100,
            "log": False
        }
    },
    "rf": {
        "n_estimators": {
            "type": "int",
            "min": 10,
            "max": 1000,
            "log": False
        },
        "ccp_alpha": {
            "type": "float",
            "min": 1e-10,
            "max": 1e-3,
            "log": True
        },
        "min_samples_split": {
            "type": "int",
            "min": 2,
            "max": 100,
            "log": False
        }
    },
    "lightgbm": {
        "n_estimators": {
            "type": "int",
            "min": 10,
            "max": 1000,
            "log": False
        },
        "num_leaves": {
            "type": "int",
            "min": 8,
            "max": 1024,
            "log": False
        },
        "max_depth": {
            "type": "int",
            "min": 3,
            "max": 10,
            "log": False
        },
        "subsample": {
            "type": "float",
            "min": 0.5,
            "max": 1.0,
            "log": False
        },
        "colsample_bytree": {
            "type": "float",
            "min": 0,
            "max": 1.0,
            "log": False
        },
        "min_split_gain": {
            "type": "float",
            "min": 1e-10,
            "max": 1e-3,
            "log": True
        },
        "reg_alpha": {
            "type": "float",
            "min": 1e-10,
            "max": 1e-3,
            "log": True
        },
        "learning_rate": {
            "type": "float",
            "min": 1e-7,
            "max": 1e-1,
            "log": True
        },
        "verbose": {
            "type": "fixed",
            "value": -1
        }
    },
    "xgboost": {
        "n_estimators": {
            "type": "int",
            "min": 10,
            "max": 1000,
            "log": False
        },
        "min_split_alpha": {
            "type": "float",
            "min": 1e-10,
            "max": 1e-3,
            "log": True
        },
        "reg_alpha": {
            "type": "float",
            "min": 1e-10,
            "max": 1e-3,
            "log": True
        },
        "learning_rate": {
            "type": "float",
            "min": 1e-7,
            "max": 1e-1,
            "log": True
        },
        "verbose": {
            "type": "fixed",
            "value": -1
        }
    },
    "svm": {
        "C": {
            "type": "float",
            "min": 1e-7,
            "max": 0.1,
            "log": True
        },
        "probability": {
            "type": "fixed",
            "value": True
        },
        "kernel": {
            "values": [
                "linear",
                "poly",
                "rbf",
                "sigmoid"
            ],
            "type": "categorical"
        },
        "max_iter": {
            "type": "fixed",
            "value": int(1e4)
        },
        "degree": {
            "condition": "kernel-poly",
            "log": False,
            "max": 7,
            "min": 2,
            "type": "int"
        }
    }
}
MACROMOLECULES_PIPELINE = {
    "name": "macromolecules_pipe",
    "aggregate": True,
    "verbose": False,
    "elements": [
        {
            "pipe-smiles-input": {
                "name": "smiles-input",
                "aggregate": False,
                "verbose": False,
                "elements": [
                    {"filter-smiles": {}}
                ]
            }
        },
        {
            "pipe-seq-input": {
                "name": "seq-input",
                "aggregate": False,
                "verbose": False,
                "elements": [
                    {"filter-smiles": {'keep_smiles': False}},
                    {"canonical-cleaner": {"substitution": "G"}},
                    {"sequence-to-smiles": {}}
                ]
            }
        }
    ]
}
SEQUENCE_PIPELINE = {
    "name": 'sequences-pipe',
    "aggregate": True,
    "verbose": False,
    "elements": [
        {
            "clean-seqs-pipe": {
                "name": "clean-seqs-pipe",
                "aggregate": False,
                "verbose": False,
                "elements": [
                    {"filter-smiles": {"keep_smiles": False}},
                    {"canonical-cleaner": {"substitution": "X"}},
                ]
            }
        },
        {
            "smiles-to-seqs-pipe": {
                "name": "smiles-to-seqs-pipe",
                "aggregate": False,
                "verbose": False,
                "elements": [
                    {"filter-smiles": {"keep_smiles": True}},
                    {"smiles-to-sequences": {}},
                    {"canonical-cleaner": {"substitution": "X"}}
                ]
            }
        }
    ]
}
MOL_REPS = {
    "chemberta-2": {
        "engine": "lm",
        "device": "cpu",
        "batch_size": 32,
        "average_pooling": True,
        'model': 'chemberta-2'
    },
    "molformer-xl": {
        'engine': "lm",
        "device": "cpu",
        "batch_size": 32,
        "average_pooling": True,
        'model': 'molformer-xl'
    },
    "peptideclm": {
        'engine': 'lm',
        'device': 'cpu',
        'batch_size': 32,
        'average_pooling': True,
        'model': "peptideclm"
    },
    "ecfp-16": {
        "engine": "fp",
        "nbits": 2048,
        "radius": 8,
        'fp': 'ecfp'
    },

}
MOL_REPS.update(
    {f'ecfp-{int(radius*2)}': {
        'engine': "fp",
        'nbits': 2048,
        'radius': radius,
        "fp": "ecfp"
    } for radius in range(1, 10, 1)}
)
MOL_REPS.update(
    {f'fcfp-{int(radius*2)}': {
        'engine': "fp",
        'nbits': 2048,
        'radius': radius,
        "fp": "fcfp"
    } for radius in range(1, 10, 1)}
)
MOL_REPS.update(
    {f'ecfp-counts-{int(radius*2)}': {
        'engine': "fp",
        'nbits': 2048,
        'radius': radius,
        "fp": "ecfp-count"
    } for radius in range(1, 10, 1)}
)
SEQ_REPS = {
    "esm2-8m": {
        'engine': 'lm',
        'device': "cpu",
        'batch_size': 32,
        "average_pooling": True,
        'model': 'esm2-8m'
    },
    "esm2-150m": {
        'engine': 'lm',
        'device': 'cpu',
        'batch_size': 32,
        'average_pooling': True,
        'model': 'esm2-150m'
    },
    "esm2-650m": {
        'engine': 'lm',
        'device': 'cpu',
        'batch_size': 16,
        'average_pooling': True,
        'model': "esm2-650m"
    },
    'prot-t5-xl': {
        'engine': 'lm',
        'device': 'cpu',
        'batch_size': 16,
        'average_pooling': True,
        'model': 'prot-t5-xl'
    },
    'prost-t5': {
        'engine': 'lm',
        'device': 'cpu',
        'batch_size': 8,
        'average_pooling': True,
        'model': 'prost-t5'
    }
}


def _is_int(text: str) -> bool:
    try:
        int(text)
        return True
    except ValueError:
        return False


def define_dataset(dataset: str, task: str, modality: str, neg: bool = False):
    if dataset.endswith('.csv') or dataset.endswith('.tsv'):
        df = pd.read_csv(dataset)
        print("These are the contents of the file you selected\n")
        print(df.head())
        print()
        columns = df.columns.tolist()
        feat_field = Prompt().select(
            "What is the number of the column with the sequences/SMILES?",
            options=columns
        )
        columns.remove(feat_field)
        if neg:
            columns_to_exclude = Prompt().checkbox(
                "What columns describe a bioactivity you would like to exclude from the negative class?",
                options=columns,
                min_selections=0
            )
            return feat_field, columns_to_exclude

        if task == 'class':
            label_field = Prompt().select(
                "What is the column containing the labels?",
                options=columns + ['Assume all entries are positive']
            )
        else:
            label_field = Prompt().select(
                "What is the column containing the labels?",
                options=columns
            )
    elif dataset.endswith('.fasta'):
        if modality != 'seqs':
            raise ValueError("FASTA is not an acceptable format for Macromolecules. Options: `csv`, `tsv`, `smi`.")
        feat_field, label_field = 'sequences', None
    return feat_field, label_field


def config_helper() -> dict:
    print()
    print("Part 1 - Define the data and preprocessing steps")
    config = {}
    task = Prompt().select(
        "What is the modelling problem you're facing?",
        options=['Classification (returning categorical value)',
                 "Regression(returnin continuous value)"]
    )
    modality = Prompt().select(
        "How do you want to define your peptides?",
        options=['Macromolecules - allows for canonical, non-canonical, and peptidomimetics',
                 'Sequences - only canonical peptides, slightly better performance']
    )
    if 'macromolecule' in modality.lower():
        modality = 'mol'
        config['pipeline'] = MACROMOLECULES_PIPELINE
    else:
        modality = 'seqs'
        config['pipeline'] = SEQUENCE_PIPELINE
    if 'class' in task.lower():
        task = 'class'
    else:
        task = 'reg'

    dataset = Prompt().input(
        "What is the path to the dataset with your data",
        validate=lambda x: osp.exists(x)
    )
    feat_field, label_field = define_dataset(dataset, task, modality)

    if task == 'class':
        print("Part 1.5 - Negative sampling")
        neg_db = Prompt().select(
            "What negative sampling strategy do you prefer?",
            options=[
                "DB of bioactive canonical peptides",
                "DB of bioactive non-canonical peptides",
                "DB of both bioactive and non-bioactive peptides",
                "Personalised DB",
                "No negative sampling"
            ]
        )
        if neg_db == 'Personalised DB':
            neg_path = Prompt().input(
                "What is the path to the dataset with your data",
                validate=lambda x: osp.exists(x)
            )

        elif neg_db == "DB of bioactive canonical peptides":
            neg_path = osp.join(
                osp.dirname(__file__), 'data', 'apml-peptipedia2.csv'
            )
            if not osp.isdir(osp.dirname(neg_path)):
                os.makedirs(osp.dirname(neg_path), exist_ok=True)
            if not osp.exists(neg_path):
                import gdown
                print("Downloading negative database...")
                FILE_ID = "189VtkbQ2bVpQlAe2UMBSzt_O4F7EyBWl"
                gdown.download(id=FILE_ID, output=neg_path, quiet=True)
        elif neg_db == 'DB of bioactive non-canonical peptides':
            neg_path = osp.join(
                osp.dirname(__file__), 'data', 'Gonzalez_2023_NC_PeptideDB.csv'
            )
            if not osp.isdir(osp.dirname(neg_path)):
                os.makedirs(osp.dirname(neg_path), exist_ok=True)
            if not osp.exists(neg_path):
                import gdown
                print("Downloading negative database...")
                FILE_ID = "1U4RXDNx_aijVDJ1oTaRKjo78Yakd3Mg4"
                gdown.download(id=FILE_ID, output=neg_path, quiet=True)

        elif neg_db == "DB of both bioactive and non-bioactive peptides":
            neg_path = osp.join(
                osp.dirname(__file__), 'data', 'apml-pep2+gonzalez.csv'
            )
            if not osp.isdir(osp.dirname(neg_path)):
                os.makedirs(osp.dirname(neg_path), exist_ok=True)
            if not osp.exists(neg_path):
                import gdown
                print("Downloading negative database...")
                FILE_ID = "189VtkbQ2bVpQlAe2UMBSzt_O4F7EyBWl"
                # url = f'https://drive.usercontent.google.com/uc?id={FILE_ID}'
                gdown.download(id=FILE_ID, output=neg_path, quiet=False)

        neg_feat_field, columns_to_exclude = define_dataset(
            neg_path, task, modality, neg=True
        )
        neg_db = {
            'path': neg_path,
            'feat_fields': neg_feat_field,
            'columns_to_exclude': columns_to_exclude,
            "verbose": False
            }

    config['databases'] = {
        'dataset': {
            'path': dataset,
            'feat_fields': feat_field,
            'label_field': label_field,
            'verbose': False
        }
    }
    if task == 'class' and neg_db != 'No negative sampling':
        config['databases']['neg_database'] = neg_db

    print("Part 2 - Define evaluation strategy")
    config['test'] = {'min_threshold': 0.1}

    if modality == 'seqs':
        sim_functions = ['needle (recommended)', 'mmseqs', 'mmseqs+prefilter (for huge datasets)']
        denominators = ['shortest', 'longest', 'n_aligned']
        sim_function = Prompt().select(
            "What alignment algorithm would you like to use?",
            options=sim_functions
        )
        denominator = Prompt().select(
            "What denominator would you like to use to compute the sequence identity?",
            options=denominators
        )
        config['test']['sim_arguments'] = {
            'data_type': 'sequence',
            'alignment_algorithm': sim_function if '+' not in sim_function else sim_function.split('+')[0],
            'denominator': denominator,
            'prefilter': 'prefilter' in sim_function,
            'min_threshold': 0.1,
            'field_name': feat_field,
            'verbose': 2
        }
    else:
        fps = ['mapc', 'ecfp', 'fcfp']
        bits = [str(int(2**v)) for v in range(8, 12, 1)]
        radii = [str(int(i)) for i in range(2, 12)]
        fp = Prompt().select(
            "What fingerprint would you like to use?",
            options=fps
        )
        bit = Prompt().select(
            "How many bits would you like the fingerprints to have? (Greater better, but more expensive)",
            options=bits
        )
        radius = Prompt().select(
            "What radius would you like to use?",
            options=radii
        )
        config['test']['sim_arguments'] = {
            'data_type': 'molecule',
            'min_threshold': 0.1,
            'sim_function': 'tanimoto' if fp == 'ecfp' else 'jaccard',
            'field_name': feat_field,
            'radius': int(radius),
            'bits': int(bit),
            'verbose': 2
        }
    partition = Prompt().select(
        "What thresholds would you like to evaluate in?",
        options=['min (AutoPeptideML v.1.0)', 'all']
    )
    part_alg = Prompt().select(
        "What partitioning algorithm would you like to use?",
        options=['ccpart', 'ccpart_random', 'graph_part'],
        default='ccpart'
    )
    config['test']['partitions'] = partition
    config['test']['algorithm'] = part_alg
    config['test']['threshold_step'] = 0.1
    config['test']['verbose'] = 2
    config['test']['filter'] = 0.185
    config['val'] = {
        'type': 'kfold',
        "k": 10,
        "random_state": 1
    }

    print("Part 3 - Define model training")
    config['train'] = {}

    learning_alg = Prompt().checkbox(
        "What models would you like to consider?",
        options=list(HP_SPACES.keys()),
        min_selections=1
    )
    model_selection = Prompt().select(
        "What model selection would you like to use?",
        options=['select', "ensemble"]
    )
    hp_search = Prompt().select(
        "What type of search for optimal hyperparameters would you like to use?",
        options=['grid', 'bayesian'],
    )
    reps = Prompt().checkbox("What representations would you like to use?",
                             options=list(MOL_REPS.keys()) if modality == 'mol'
                             else list(SEQ_REPS.keys()), min_selections=1)
    acc = Prompt().select("Which accelerator would you like to use to compute the representations?",
                          options=['cpu', "cuda", "mps"])
    hp_search = hp_search if hp_search != 'bayesian' else 'optuna'
    if hp_search == 'optuna':
        n_steps = Prompt().input(
            "How many steps for optimisation would you like to conduct?",
            default=100,
            validate=_is_int
        )
        patience = Prompt().input(
            "What patience would you like EarlyStopping to have?",
            validate=_is_int
        )
    n_jobs = Prompt().input(
        "How many parallel jobs do you want to run?",
        default=cpu_count(),
        validate=_is_int
    )
    config['train']['task'] = task
    config['train']['optim_strategy'] = {
        'trainer': hp_search,
        'n_steps': int(n_steps) if hp_search == 'optuna' else None,
        'direction': "maximize",
        'task': task,
        'metric': 'pcc' if task == 'reg' else 'mcc',
        'partition': 'random',
        'n_jobs': int(n_jobs),
        'patience': int(patience) if hp_search == 'optuna' else None
    }
    config['train']['hspace'] = {'representations': reps}
    config['train']['hspace']['models'] = {
        'type': model_selection,
        'elements': {model: HP_SPACES[model] for model in learning_alg},
    }
    config['representation'] = {
        'verbose': True,
        'elements': [
            {
                r: MOL_REPS[r] if modality == 'mol' else SEQ_REPS[r]
            } for r in reps
        ]
    }
    for idx, element in enumerate(config['representation']['elements']):
        name = list(element.keys())[0]
        if config['representation']['elements'][idx][name]['engine'] != 'lm':
            continue
        config['representation']['elements'][idx][name]['device'] = acc
    path = Prompt().input(
        "Where do you want to save the experiment results?",
        validate=lambda x: not osp.isdir(x)
    )
    config['outputdir'] = path
    path = osp.join(path, 'config.yml')
    os.makedirs(path, exist_ok=True)
    yaml.safe_dump(config, open(path, 'w'), indent=2)
    return path
