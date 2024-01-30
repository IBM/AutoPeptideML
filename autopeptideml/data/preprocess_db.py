import argparse
import os
import shutil
import time
from multiprocessing import cpu_count

import gdown
import pandas as pd
from pandarallel import pandarallel

from autopeptideml.data.residues import is_canonical


def welcome():
    print('-' * len('| Welcome to the AutoPeptideML Installation Tool |'))
    print('| Welcome to the AutoPeptideML Installation Tool |')
    print('-' * len('| Welcome to the AutoPeptideML Installation Tool |'))
    print('This tool will download and preprocess the Peptipedia Database')
    print('The process may take several minutes')
    print('Please wait...')
    print()

def goodbye():
    print('Installation successful')
    print('Now you can enjoy AutoPeptideML')
    print()

def download_db(url: str, **kwargs):
    """
    Download Peptipedia Database 

    Parameters
    ----------
    url : str
        URL of the Peptipedia Databas0

    Returns
    -------
    file_path : str
        Path to the downloaded Peptipedia Database

    Notes
    -----
    The Peptipedia Database is hosted on Google Drive.
    """
    start = time.time()
    print('Downloading Peptipedia Database')
    data_path = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    output = os.path.join(data_path, 'peptipedia.zip')
    gdown.download(url, output, quiet=False)

    # Clean directory in case of a previous installation
    if os.path.isdir(os.path.join(data_path, 'peptipedia')):
        shutil.rmtree(os.path.join(data_path, 'peptipedia'))

    os.mkdir(os.path.join(data_path, 'peptipedia'))
    shutil.unpack_archive(output, os.path.join(data_path, 'peptipedia'))

    file_path = os.path.join(data_path, 'peptipedia.csv')
    shutil.unpack_archive(os.path.join(data_path, 'peptipedia', 'peptipedia_csv.zip'), data_path)

    # Remove temporary files
    shutil.rmtree(os.path.join(data_path, 'peptipedia'))
    os.remove(os.path.join(data_path, 'peptipedia.zip'))

    # Remove any temporary files that may remain from a previous failed installation
    for file in os.listdir(data_path):
        if file[-3:] == 'tmp':
            os.remove(os.path.join(data_path, file))

    print('Download complete in {:.2f} seconds'.format(time.time() - start))

    return file_path

def partition_db(df: pd.DataFrame, threads: int, **kwargs):
    max_length = df.sequence.parallel_map(len).max()
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'peptipedia')
    os.mkdir(path)

    for length in range(5, max_length+5, 5):
        df_length = df[(df.sequence.parallel_map(len) >= length-5) & (df.sequence.parallel_map(len) < length)]
        if len(df_length) == 0:
            continue
        lengt_path = os.path.join(path, f'peptipedia_{length-5}-{length}.csv')
        df_length.to_csv(lengt_path, index=False)

def preprocess_db(path: str, threads: int, **kwargs):
    """
    Preprocess Peptipedia Database

    Parameters
    ----------
    path : str
        Path to the Peptipedia Database
    threads : int
        Number of threads to use
    """
    print()
    print('Processing Peptipedia Database')
    start = time.time()
    df = pd.read_csv(path)
    df.drop_duplicates(subset=['sequence'], inplace=True, ignore_index=True)
    # df = df.parallel_apply(lambda x: x.iloc[3:].astype(int), axis=1)
    df = df[df.parallel_apply(lambda x: x.iloc[3:].sum() > 0, axis=1)]
    df = df[df.sequence.parallel_map(is_canonical)]
    df = df.reset_index(drop=True)

    with open(os.path.join('/'.join(path.split('/')[:-1]), 'bioactivities.txt'), 'w') as file:
        for column in df:
            if column not in ['sequence', 'idpeptide', 'is_aa_seq']:
                file.write(f'{column}\n')

    partition_db(df, threads, **kwargs)
    print('Processing complete in {:.2f} seconds'.format(time.time() - start))
    print()

def main():
    parser = argparse.ArgumentParser(description='AutoPeptideML Installation Tool')
    parser.add_argument('--threads', type=int, default=cpu_count(),
                        help='Number of threads to use')
    parser.add_argument('--url', type=str, default='https://drive.google.com/uc?id=1x3yHNl8k5teHlBI2FMgl966o51s0T8i_',
                        help='URL of the Peptipedia Database')
    args = parser.parse_args()
    args = vars(args)
    welcome()
    pandarallel.initialize(nb_workers=args['threads'], verbose=1)
    path = download_db(**args)
    preprocess_db(path, **args)
    os.remove(path)
    goodbye()


def prepare_db():
    threads = cpu_count()
    url = 'https://drive.google.com/uc?id=1x3yHNl8k5teHlBI2FMgl966o51s0T8i_'
    welcome()
    pandarallel.initialize(nb_workers=threads, verbose=1)
    path = download_db(threads=threads, url=url)
    preprocess_db(path, threads=threads)
    os.remove(path)
    goodbye()
