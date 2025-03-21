import os
import os.path as osp
import urllib.error
import pkg_resources
import shutil
import time
import urllib
import urllib.request as request
import zipfile

import pandas as pd
import typer

import datamol as dm
import rdkit.Chem as Chem
from rdkit.Chem import rdmolfiles
from pyPept.converter import Converter
from pepfunn.sequence import peptideFromSMILES
from pqdm.processes import pqdm


SPECIAL_1 = ['ac-', 'deca-', 'glyco-', 'medl-', 'Mono21-', 'Mono22-']
SPECIAL_2 = ['-pip']
MONOMERS = SPECIAL_1 + SPECIAL_2
CANONICAL = {
    "ALA": "A", "ASP": "D", "GLU": "E", "PHE": "F", "HIS": "H",
    "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "GLY": "G",
    "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
    "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y", "CYS": "C"
}


def is_canonical(sequence: str):
    if not (len(sequence) > 0):
        return False
    for char in sequence:
        if char not in CANONICAL.values():
            return False
    return True


def pepseqres2biln(biln: str) -> str:
    new_biln = biln
    for monomer in MONOMERS:
        monomer = monomer
        if (f'-{monomer}-' in biln or f'({monomer}-' in biln or
           f'-{monomer}(' in biln):
            new_biln = new_biln.replace(monomer, f'[{monomer}]')
        elif monomer in SPECIAL_1 and f'{monomer}-' in biln:
            new_biln = new_biln.replace(monomer, f'[{monomer}]')
        elif monomer in SPECIAL_2 and f'-{monomer}' in biln:
            new_biln = new_biln.replace(monomer, f'[{monomer}]')

    for monomer in CANONICAL:
        new_biln = new_biln.replace(monomer, CANONICAL[monomer])
    return new_biln


def prepare_resources():
    global MONOMERS
    temporal = []
    file_path = pkg_resources.resource_filename('pepfunn', 'data/property.txt')

    with open(file_path, 'r') as file:
        data = file.read()

    lines = data.split('\n')
    for line in lines:
        if line:
            temporal.append(line.split()[0])

    for mon in temporal:
        if '-' in mon:
            MONOMERS.append(mon)


def fasta2smiles(seq: str) -> str:
    mol = Chem.MolFromSequence(seq)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def fasta2biln(seq: str) -> str:
    biln = '-'.join(seq)
    new_biln = biln
    for monomer in MONOMERS:
        monomer = monomer

        if (f'-{monomer}-' in biln or f'({monomer}-' in biln or
           f'-{monomer}(' in biln):
            new_biln = new_biln.replace(monomer, f'[{monomer}]')
        elif monomer in SPECIAL_1 and f'{monomer}-' in biln:
            new_biln = new_biln.replace(monomer, f'[{monomer}]')
        elif monomer in SPECIAL_2 and f'-{monomer}' in biln:
            new_biln = new_biln.replace(monomer, f'[{monomer}]')
        elif monomer in CANONICAL:
            new_biln = new_biln.replace(monomer, CANONICAL[monomer])
    for monomer in CANONICAL:
        new_biln = new_biln.replace(monomer, CANONICAL[monomer])

    return new_biln


def download_downstream_data(data_path: str) -> None:
    """
    Download the downstream data for the project.

    Args:
        data_path (str): The path where the data will be downloaded.

    Returns:
        None
    """
    prepare_resources()
    if osp.exists(data_path):
        raise RuntimeError(f'Warning! Path: {data_path} already exists')
    else:
        os.makedirs(data_path)

    print('Downloading Canonical Cell Penetration dataset...')
    status = download_c_cpp(data_path)
    if status:
        print('Canonical Cell Penetration dataset downloaded succesfully!')
    else:
        print('There has been a problem with the download, omitting.')

    print('Downloading Non-canonical Cell Penetration dataset...')
    status = download_nc_cpp(data_path)
    if status:
        print('Non-canonical Cell Penetration dataset downloaded succesfully!')
    else:
        print('There has been a problem with the download, omitting.')

    print('Downloading Canonical binding dataset...')
    status = download_c_binding(data_path)
    if status:
        print('Canonical binding dataset downloaded succesfully!')
    else:
        print('There has been a problem with the download, omitting.')

    print('Downloading Non-canonical binding dataset...')
    status = download_nc_binding(data_path)
    if status:
        print('Non-canonical binding dataset downloaded succesfully!')
    else:
        print('There has been a problem with the download, omitting.')

    print('Downloading Non-canonical Antiviral dataset...')
    status = download_nc_antiviral(data_path)
    if status:
        print('Non-canonical antiviral dataset downloaded succesfully!')
    else:
        print('There has been a problem with the download, omitting.')

    print("Downloading Canonical Antiviral...")
    status = download_c_antiviral(data_path)
    if status:
        print("Canonical Antiviral dataset downloaded succesfully!")
    else:
        print("There has been a problem with the download, omitting.")

    print('Downloading Non-canonical Antibacterial dataset...')
    status = download_nc_antibacterial(data_path)
    if status:
        print('Non-canonical antimicrobial dataset downloaded succesfully!')
    else:
        print('There has been a problem with the download, omitting.')

    print("Downloading Canonical Antibacterial...")
    status = download_c_antibacterial(data_path)
    if status:
        print("Canonical Antimicrobial dataset downloaded succesfully!")
    else:
        print("There has been a problem with the download, omitting.")


def download_nc_antiviral(data_path: str) -> bool:
    url = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/2zhgy9ggdv-2.zip'
    tmp_path = osp.join(data_path, 'PharmaPeptides.zip')
    extracted_path = osp.join(data_path, "Database of Peptides with Potential for Pharmacological Intervention in Human Pathogen Molecular Targets")
    out_path = osp.join(data_path, 'nc-antiviral.csv')
    try:
        request.urlretrieve(url, tmp_path)
    except urllib.error.URLError:
        return False
    with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    os.remove(tmp_path)
    positive = []
    negative = []
    subdirs = ['Antiparasitic peptides', 'Antifungal peptides', 'Antibacterial peptides', 'Antiviral peptides']
    for subdir in subdirs:
        subdir_path = osp.join(extracted_path, subdir, "smi")
        for file in os.listdir(subdir_path):
            with open(osp.join(subdir_path, file)) as fi:
                txt = fi.readlines()[0].rstrip()
            if subdir != 'Antiviral peptides':
                negative.append({
                    'SMILES': txt,
                    'labels': 0})
            else:
                positive.append({
                    'SMILES': txt,
                    'labels': 1})
    shutil.rmtree(extracted_path)
    df_pos = pd.DataFrame(positive)
    df_neg = pd.DataFrame(negative).sample(len(df_pos), replace=False,
                                           random_state=1)
    df = pd.concat([df_pos, df_neg])
    smiles = df['SMILES']
    # df['BILN'] = pqdm(smiles, peptideFromSMILES, n_jobs=10,
    #                   exception_behaviour='immediate')
    df.to_csv(out_path, index=False)
    return True


def download_c_antiviral(data_path: str) -> bool:
    tmp_path = osp.join(data_path, 'peptidebenchmarks.tar.gz')
    tmp_dir = osp.join(data_path, 'PeptideBenchmarks')
    url = "https://drive.google.com/u/0/uc?id=1UmDu773CdkBFqkitK550uO6zoxhU1bUB&export=download"
    try:
        request.urlretrieve(url, tmp_path)
    except urllib.error.URLError:
        return False
    shutil.unpack_archive(tmp_path, data_path)
    os.remove(tmp_path)
    df1 = pd.read_csv(osp.join(tmp_dir, 'AV', 'splits', 'train.csv'))
    df1 = df1.drop_duplicates('sequence')
    df2 = pd.read_csv(osp.join(tmp_dir, 'AV', 'splits', 'test.csv'))
    df2 = df2.drop_duplicates('sequence')
    df = pd.concat([df1, df2])
    df['SMILES'] = df.sequence.map(fasta2smiles)
    df['BILN'] = df.sequence.map(fasta2biln)
    df['labels'] = df.Y
    shutil.rmtree(tmp_dir)
    print(len(df))
    df = df[df.sequence.map(len) <= 50]
    print(len(df))
    df.to_csv(osp.join(data_path, 'c-antiviral.csv'), index=False)
    return True


def download_c_antibacterial(data_path: str) -> bool:
    tmp_path = osp.join(data_path, 'peptidebenchmarks.tar.gz')
    tmp_dir = osp.join(data_path, 'PeptideBenchmarks')
    url = "https://drive.google.com/u/0/uc?id=1UmDu773CdkBFqkitK550uO6zoxhU1bUB&export=download"
    try:
        request.urlretrieve(url, tmp_path)
    except urllib.error.URLError:
        return False
    shutil.unpack_archive(tmp_path, data_path)
    os.remove(tmp_path)
    df1 = pd.read_csv(osp.join(tmp_dir, 'AB', 'splits', 'train.csv'))
    df1 = df1.drop_duplicates('sequence')
    df2 = pd.read_csv(osp.join(tmp_dir, 'AB', 'splits', 'test.csv'))
    df2 = df2.drop_duplicates('sequence')
    df = pd.concat([df1, df2])
    df['SMILES'] = df.sequence.map(fasta2smiles)
    df['BILN'] = df.sequence.map(fasta2biln)
    df['labels'] = df.Y
    shutil.rmtree(tmp_dir)
    print(len(df))
    df = df[df.sequence.map(len) <= 50]
    print(len(df))
    df.to_csv(osp.join(data_path, 'c-antibacterial.csv'), index=False)
    return True


def download_nc_antibacterial(data_path: str) -> bool:
    out_path = osp.join(data_path, 'nc-antibacterial.csv')
    proto_df = []
    url = 'https://webs.iiitd.edu.in/raghava/antimpmod'
    for file in ['pos_train', 'neg_train', 'pos_test', 'neg_test']:
        d_url = f'{url}/{file}.zip'
        filepath = osp.join(data_path, f'{file}.zip')

        try:
            request.urlretrieve(d_url, filepath)
        except urllib.error.URLError:
            return False
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        os.remove(filepath)
        extracted = osp.join(data_path, file)
        dm.disable_rdkit_log()

        for pdb_file in os.listdir(extracted):
            mol = rdmolfiles.MolFromPDBFile(
                osp.join(extracted, pdb_file), sanitize=False, removeHs=False
            )
            mol = dm.fix_mol(mol)
            mol = dm.fix_valence(mol)
            mol = dm.remove_hs(mol)
            if mol is None:
                raise ValueError(f"File: {file} has not generated a valid molecule.")

            proto_df.append({
                'SMILES': dm.to_smiles(mol),
                'labels': 1 if 'pos' in file else 0,
                'original_split': 1 if 'test' in file else 0,
            })
        shutil.rmtree(extracted)
    df = pd.DataFrame(proto_df)
    # smiles = df['SMILES']
    # df['BILN'] = pqdm(smiles, peptideFromSMILES, n_jobs=10,
    #                   exception_behaviour='immediate')
    df.to_csv(out_path)
    return True


def download_c_cpp(data_path: str) -> bool:
    out_path = osp.join(data_path, 'c-cpp.csv')
    url = 'https://raw.githubusercontent.com/zhangruochi/pepland/master/data/eval/c-CPP.txt'
    try:
        request.urlretrieve(url, out_path)
    except urllib.error.URLError:
        return False
    df = pd.read_csv(out_path, header=None, names=['sequence', 'labels'])
    df['BILN'] = df['sequence'].apply(fasta2biln)
    df['SMILES'] = df['sequence'].apply(fasta2smiles)
    df = df[df.sequence.map(is_canonical)]
    df = df.dropna().reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return True


def download_nc_cpp(data_path: str) -> bool:
    out_path = osp.join(data_path, 'nc-cpp.csv')
    url = 'https://raw.githubusercontent.com/zhangruochi/pepland/master/data/eval/nc-CPP.csv'
    try:
        request.urlretrieve(url, out_path)
    except urllib.error.URLError:
        return False
    df = pd.read_csv(out_path)
    df['labels'] = df['PAMPA']
    neg_df = df[df.labels < -9.5].copy()
    neg_df['labels'] = 0
    pos_df = df[(df.labels >= -6.5) & (df.labels <= 4.5)].sample(len(neg_df), random_state=1).copy()
    pos_df['labels'] = 1
    df = pd.concat([neg_df, pos_df])
    df = df[['SMILES', 'HELM', 'labels']]
    df['BILN'] = df['SMILES'].map(peptideFromSMILES)
    df.dropna(inplace=True)
    df.to_csv(out_path, index=False)
    return True


def download_nc_binding(data_path: str) -> bool:
    out_path = osp.join(data_path, 'nc-binding.csv')
    url = "https://raw.githubusercontent.com/zhangruochi/pepland/master/data/eval/nc-binding.csv"
    try:
        request.urlretrieve(url, out_path)
    except urllib.error.URLError:
        return False
    df = pd.read_csv(out_path)
    df['SMILES'] = df['Merge_SMILES']
    df['BILN'] = df['pep_SEQRES'].apply(pepseqres2biln)
    df['labels'] = df['affinity']
    df = df[['seq1', 'SMILES', 'BILN', 'labels']]
    df.dropna(inplace=True)
    df.to_csv(out_path, index=False)
    return True


def download_c_binding(data_path: str) -> bool:
    out_path = osp.join(data_path, 'c-binding.csv')
    url = "https://raw.githubusercontent.com/zhangruochi/pepland/master/data/eval/c-binding.csv"
    try:
        request.urlretrieve(url, out_path)
    except urllib.error.URLError:
        return False
    df = pd.read_csv(out_path)
    df['SMILES'] = df['Merge_SMILES']
    df['BILN'] = df['pep_SEQRES'].apply(pepseqres2biln)
    df['labels'] = df['affinity']
    df = df[df['BILN'].map(lambda x:  max([len(a) for a in x.split('-')]) == 1)]
    df['sequence'] = df['BILN'].map(lambda x: ''.join(x.split('-')))
    df = df[df.sequence.map(is_canonical)]
    df = df[['seq1', 'sequence', 'SMILES', 'BILN', 'labels']]
    df.dropna(inplace=True)
    df.to_csv(out_path, index=False)
    return True


if __name__ == '__main__':
    typer.run(download_downstream_data)
