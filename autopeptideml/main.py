import argparse
import json
from multiprocessing import cpu_count
import os

import pandas as pd

from .autopeptideml import AutoPeptideML
from .utils.embeddings import RepresentationEngine


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--outputdir', type=str, default='apml_result')

    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=cpu_count())
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--plm', type=str, default='esm2-8m',
                        help='PLM for computing peptide representations. Check GitHub Repository for available options.')
    parser.add_argument('--plm_batch_size', type=int, default=12)
    parser.add_argument('--config', type=str, default='default_config')

    parser.add_argument('--autosearch', type=str, default='auto',
                        help='Whether to search for negative peptides.')
    parser.add_argument('--autosearch_tags', type=str, default='',
                        help='Comma-separated list of positive tags to exclude from autosearch.')
    parser.add_argument('--autosearch_proportion', type=float, default=2,
                        help='Negative:positive proportion. Float number.')

    parser.add_argument('--balance', type=str, default='False',
                        help='Whether to oversample the underrepresented class.')

    parser.add_argument('--test_partition', type=str, default='True',
                        help='Whether to divide dataset in train/test splits.')
    parser.add_argument('--test_threshold', type=float, default=0.3)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--test_alignment', type=str, default='mmseqs+prefilter')
    parser.add_argument('--splits', type=str, default=None)

    parser.add_argument('--val_partition', type=str, default='True',
                        help='Whether to divide dataset in train/val folds.')
    parser.add_argument('--val_method', type=str, default='random')
    parser.add_argument('--val_alignment', type=str, default='mmseqs+prefilter')
    parser.add_argument('--val_threshold', type=float, default=0.5)
    parser.add_argument('--val_n_folds', type=int, default=10)
    parser.add_argument('--folds', type=str, default=None)

    return parser.parse_args()

def parse_cli_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--ensemble', type=str,
                        help='Path to dir with previous APML results.')

    parser.add_argument('--outputdir', type=str, default='apml_predictions')

    parser.add_argument('--verbose', type=str, default='True')
    parser.add_argument('--threads', type=int, default=cpu_count())

    parser.add_argument('--plm', type=str, default='esm2-8m',
                        help='PLM for computing peptide representations. Check GitHub Repository for available options.')
    parser.add_argument('--plm_batch_size', type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_cli()

    os.makedirs(args.outputdir, exist_ok=True)
    apml_config_path = os.path.join(
        args.outputdir, 'apml_config.json'
    )
    json.dump(vars(args), open(apml_config_path, 'w'), indent=4)

    re = RepresentationEngine(args.plm, args.plm_batch_size)
    apml = AutoPeptideML(args.verbose, args.threads, args.seed)

    if args.dataset != 'None':
        df = apml.curate_dataset(args.dataset, args.outputdir)
        if 'id' not in df.columns:
            df['id'] = df.index

    if (args.autosearch == 'auto' and len(df[df.Y == 0]) < 1 or
       args.autosearch == 'True'):
        df = apml.autosearch_negatives(
            df,
            args.autosearch_tags.split(','),
            args.autosearch_proportion
        )
    if args.balance == 'True':
        df = apml.balance_samples(df)

    if args.test_partition == 'True' and args.splits is None:
        datasets = apml.train_test_partaition(
            df,
            args.test_threshold,
            args.test_size,
            args.test_alignment,
            os.path.join(args.outputdir, 'splits')
        )
    else:
        datasets = {
            'train': pd.read_csv(os.path.join(args.splits, 'train.csv')),
            'test': pd.read_csv(os.path.join(args.splits, 'test.csv'))
        }

    if args.val_partition == 'True' and args.folds is None:
        folds = apml.train_val_partition(
            datasets['train'],
            args.val_method,
            args.val_threshold,
            args.val_alignment,
            args.val_n_folds,
            os.path.join(args.outputdir, 'folds')
        )
    else:
        folds = [
            {'train': pd.read_csv(os.path.join(args.folds, f'train_{i}.csv')),
             'val': pd.read_csv(os.path.join(args.folds, f'val_{i}.csv'))}
             for i in range(args.val_n_folds)
        ]

    id2rep = apml.compute_representations(datasets, re)
    if not args.config.endswith('.json'):
        args.config = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'data', 'configs', args.config + '.json'
        )

    model = apml.hpo_train(
        json.load(open(args.config)),
        datasets['train'],
        id2rep,
        folds,
        args.outputdir,
    )
    results = apml.evaluate_model(
        model, 
        datasets['test'],
        id2rep,
        args.outputdir
    )
    if args.verbose is True:
        print(results)

def predict():
    args = parse_cli_predict()

    re = RepresentationEngine(args.plm, args.plm_batch_size)    
    apml = AutoPeptideML(args.verbose, args.threads, 1)
    df = apml.curate_dataset(args.dataset, args.outputdir)
    apml.predict(df, re, args.ensemble, args.outputdir)
