import os.path as osp
import yaml

from copy import deepcopy

import pandas as pd

from ItsPrompt.prompt import Prompt

from .db.negative_sampling import get_neg_db
from .train.architectures import ALL_MODELS


def _is_int(text: str) -> bool:
    try:
        int(text)
        return True
    except ValueError:
        return False


def define_dataset(dataset: str, task: str, neg: bool = False):
    if dataset.endswith('.csv') or dataset.endswith('.tsv'):
        df = pd.read_csv(dataset)
        print("These are the contents of the file you selected\n")
        print(df.head())
        print()
        columns = df.columns.tolist()
        feat_field = Prompt().select(
            "What is the column with the sequences/SMILES?",
            options=columns
        )
        columns.remove(feat_field)
        if neg:
            columns_to_exclude = Prompt().checkbox(
                "What columns describe a bioactivity you would like to exclude from the negative class?",
                options=columns(),
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
        feat_field, label_field = 'sequences', None
    else:
        if neg:
            n_df = get_neg_db(target_db=dataset, verbose=False)
            print("These are the contents of the file you selected\n")
            print(n_df.head())
            print()
            columns = n_df.columns.tolist()
            feat_field = None
            columns_to_exclude = Prompt().checkbox(
                "What columns describe a bioactivity you would like to exclude from the negative class?",
                options=sorted(n_df.columns.tolist()),
                min_selections=0
            )
            return feat_field, columns_to_exclude

    return feat_field, label_field


def config_helper(config_path: str) -> dict:
    print()
    print("Part 1 - Define the data and preprocessing steps")
    config = {}
    task = Prompt().select(
        "What is the modelling problem you're facing?",
        options=['Classification (returning categorical value)',
                 "Regression(returnin continuous value)"]
    )
    if 'class' in task.lower():
        task = 'class'
    else:
        task = 'reg'

    config['pipeline'] = 'to-smiles'
    config['task'] = task

    dataset = Prompt().input(
        "What is the path to the dataset with your data",
        validate=lambda x: osp.exists(x)
    )
    dataset = osp.abspath(dataset)
    feat_field, label_field = define_dataset(dataset, task)

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
            neg_path = osp.abspath(neg_path)
        elif neg_db == "DB of bioactive canonical peptides":
            neg_path = 'canonical'
        elif neg_db == 'DB of bioactive non-canonical peptides':
            neg_path = 'non-canonical'
        elif neg_db == "DB of both bioactive and non-bioactive peptides":
            neg_path = 'both'
        else:
            print(neg_db)

        neg_feat_field, columns_to_exclude = define_dataset(
            neg_path, task, neg=True
        )
        neg_db = {
            'path': neg_path,
            'feat-fields': neg_feat_field,
            'activities-to-exclude': columns_to_exclude
            }

    config['datasets'] = {
        'main': {
            'path': dataset,
            'feat-fields': feat_field,
            'label-field': label_field
        },
    }
    if task == 'class' and neg_db != 'No negative sampling':
        config['datasets']['neg-db'] = neg_db

    print("Part 3 - Define model training")

    models = Prompt().checkbox(
        "What models would you like to consider?",
        options=ALL_MODELS,
        min_selections=1
    )
    reps = Prompt().checkbox("What representations would you like to use?",
                             options=['ecfp', 'chemberta-2', 'molformer-xl',
                                      'peptide-clm', 'esm2-8m'],
                             min_selections=1)
    acc = Prompt().select("Which accelerator would you like to use to compute the representations?",
                          options=['cpu', "cuda", "mps"])

    n_steps = Prompt().input(
        "How many steps for optimisation would you like to conduct?",
        default=100,
        validate=_is_int
    )
    config.update({
        'n-trials': int(n_steps),
        'direction': "maximize",
        'task': task,
        'metric': 'spcc' if task == 'reg' else 'mcc',
        'split-strategy': 'min',
        'device': acc,
        'reps': reps,
        'models': models,
        'n-jobs': -1
    })
    yaml.safe_dump(deepcopy(config), open(config_path, 'w'))
    return config
