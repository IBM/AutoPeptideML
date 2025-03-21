import json
import os

import numpy as np
import pandas as pd
import typer

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map


def calculate_esm(dataset: str, model: str):
    from autopeptideml.reps.lms import RepEngineLM

    re = RepEngineLM(model, average_pooling=True)
    re.move_to_device('mps')
    out_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps', f'{model}_{dataset}.json'
    )
    os.makedirs((os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps')), exist_ok=True)
    if os.path.exists(out_path):
        return json.load(open(out_path))
    df = pd.read_csv(os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'downstream_data', f'{dataset}.csv'
    ))
    fp = re.compute_reps(df.sequence.tolist(), average_pooling=True, batch_size=64 if re.get_num_params() < 1e8 else 16)
    fp = [f.tolist() for f in fp]
    json.dump(fp, open(os.path.join(out_path), 'w'))


def calculate_ecfp(dataset: str):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    out_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps', f'ecfp_{dataset}.json'
    )
    os.makedirs((os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps')), exist_ok=True)
    if os.path.exists(out_path):
        return json.load(open(out_path))
    df = pd.read_csv(os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'downstream_data', f'{dataset}.csv'
    ))
    fpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=8, fpSize=2_048
    )

    def _get_fp(smile: str):
        mol = Chem.MolFromSmiles(smile)
        fp = fpgen.GetFingerprintAsNumPy(mol).astype(np.int8)
        return fp

    fps = thread_map(
        _get_fp, df['SMILES'], max_workers=8
    )
    fps = np.stack(fps).tolist()
    json.dump(fps, open(os.path.join(out_path), 'w'))


def calculate_chemberta(dataset: str):
    import transformers as hf
    import torch

    device = 'mps'
    batch_size = 32
    out_path = os.path.join(os.path.dirname(__file__),
        '..', '..', 'reps', f'chemberta_{dataset}.json')
    if os.path.exists(out_path):
        return np.array(json.load(open(out_path)))
    os.makedirs((os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps')), exist_ok=True)
    df = pd.read_csv(os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'downstream_data', f'{dataset}.csv'
    ))
    tokenizer = hf.AutoTokenizer.from_pretrained(
        'DeepChem/ChemBERTa-77M-MLM', trust_remote_code=True
    )
    model = hf.AutoModel.from_pretrained('DeepChem/ChemBERTa-77M-MLM',
                                         trust_remote_code=True)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params / 1e6 < 1e3:
        print(f'Number of model parameters are: {n_params/1e6:.1f} M')
    else:
        print(f'Number of model parameters are: {n_params/1e9:.1f} B')
    smiles = df['SMILES'].tolist()
    batched = [smiles[i:i+batch_size] for i in
               range(0, len(smiles), batch_size)]
    fps = []
    for batch in tqdm(batched):
        input_ids = tokenizer(batch, return_tensors='pt',
                              padding='longest', truncation=True).to(device)
        with torch.no_grad():
            vector = model(**input_ids).last_hidden_state
            mask = input_ids['attention_mask']
            for i in range(mask.shape[0]):
                length = mask[i].sum()
                fps.append(vector[i, :length].mean(0).detach().cpu().tolist())
    json.dump(fps, open(os.path.join(out_path), 'w'))
    return np.array(fps)


def calculate_molformer(dataset: str):
    import transformers as hf
    import torch


    device = 'mps'
    batch_size = 32
    out_path = os.path.join(os.path.dirname(__file__),
        '..', '..', 'reps', f'molformer_{dataset}.json')
    if os.path.exists(out_path):
        return np.array(json.load(open(out_path)))
    os.makedirs((os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps')), exist_ok=True)
    df = pd.read_csv(os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'downstream_data', f'{dataset}.csv'
    ))
    tokenizer = hf.AutoTokenizer.from_pretrained(
        'ibm/MoLFormer-XL-both-10pct', trust_remote_code=True
    )
    model = hf.AutoModel.from_pretrained('ibm/MoLFormer-XL-both-10pct',
                                         trust_remote_code=True)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params / 1e6 < 1e3:
        print(f'Number of model parameters are: {n_params/1e6:.1f} M')
    else:
        print(f'Number of model parameters are: {n_params/1e9:.1f} B')
    smiles = df['SMILES'].tolist()
    batched = [smiles[i:i+batch_size] for i in
               range(0, len(smiles), batch_size)]
    fps = []
    for batch in tqdm(batched):
        input_ids = tokenizer(batch, return_tensors='pt',
                              padding='longest').to(device)
        with torch.no_grad():
            vector = model(**input_ids).last_hidden_state
            mask = input_ids['attention_mask']
            for i in range(mask.shape[0]):
                length = mask[i].sum()
                fps.append(vector[i, :length].mean(0).detach().cpu().tolist())
    json.dump(fps, open(os.path.join(out_path), 'w'))
    return np.array(fps)


def calculate_pepclm(dataset: str):
    from utils.pepclm_tokenizer import SMILES_SPE_Tokenizer
    import transformers as hf
    import torch

    device = 'mps'
    batch_size = 8
    out_path = os.path.join(os.path.dirname(__file__),
        '..', '..', 'reps', f'pepclm_{dataset}.json')
    if os.path.exists(out_path):
        return json.load(open(out_path))
    os.makedirs((os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps')), exist_ok=True)
    df = pd.read_csv(os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'downstream_data', f'{dataset}.csv'
    ))
    tokenizer = SMILES_SPE_Tokenizer(
        os.path.join(os.path.dirname(__file__), 'utils',
                     'tokenizer', 'new_vocab.txt'),
        os.path.join(os.path.dirname(__file__), 'utils',
                     'tokenizer', 'new_splits.txt')
    )
    model = hf.AutoModel.from_pretrained('aaronfeller/PeptideCLM-23M-all',
                                         trust_remote_code=True)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params / 1e6 < 1e3:
        print(f'Number of model parameters are: {n_params/1e6:.1f} M')
    else:
        print(f'Number of model parameters are: {n_params/1e9:.1f} B')
    smiles = df['SMILES'].tolist()
    batched = [smiles[i:i+batch_size] for i in
               range(0, len(smiles), batch_size)]
    fps = []
    for batch in tqdm(batched):
        input_ids = tokenizer(batch, return_tensors='pt',
                              padding='longest').to(device)
        with torch.no_grad():
            vector = model(**input_ids).last_hidden_state
            mask = input_ids['attention_mask']
            for i in range(mask.shape[0]):
                length = mask[i].sum()
                fps.append(vector[i, :length].mean(0).detach().cpu().tolist())
    json.dump(fps, open(os.path.join(out_path), 'w'))


def calculate_pepfunnfp(dataset: str):
    from pepfunn.similarity import monomerFP

    out_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps', f'pepfunn_{dataset}.json'
    )
    os.makedirs((os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps')), exist_ok=True)
    if os.path.exists(out_path):
        return json.load(open(out_path))
    df = pd.read_csv(os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'downstream_data', f'{dataset}.csv'
    ))
    n_bits = 2048

    def _get_fp(smile: str):
        try:
            fp, dict_fp = monomerFP(smile, radius=2, nBits=n_bits,
                                    add_freq=True,
                                    property_lib='property_ext.txt')
        except ValueError:
            if 'X' in smile:
                smile = smile.replace('X', 'G')
                fp = _get_fp(smile)
            else:
                print(smile)
                return np.zeros((n_bits,))
        except TypeError:
            return np.zeros((n_bits,))
        return np.array(fp)

    fps = thread_map(
        _get_fp, df['BILN'], max_workers=8
    )
    counter = len([a for a in fps if a.sum() == 0])
    print('Faulty Pepfunn: ', counter)
    # fps = [np.zeros((2048,)) if a is None else a for a in fps]
    fps = np.stack(fps).tolist()
    json.dump(fps, open(os.path.join(out_path), 'w'))


def calculate_pepland(dataset: str):
    from utils.pepland_utils.inference import run

    out_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps', f'pepland_{dataset}.json'
    )
    os.makedirs((os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'reps')), exist_ok=True)
    df = pd.read_csv(os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'downstream_data', f'{dataset}.csv'
    ))

    embds = run(df.SMILES.tolist(), 1)
    embds = embds.tolist()
    json.dump(embds, open(out_path, 'w'))


def main(dataset: str, rep: str):
    if rep == 'ecfp':
        print('Calculating ECFP representations...')
        calculate_ecfp(dataset)
    elif rep == 'molformer':
        print('Calculating MolFormer-XL representations...')
        calculate_molformer(dataset)
    elif rep == 'chemberta':
        print('Calculating ChemBERTa-2 77M MLM representations')
        calculate_chemberta(dataset)
    elif rep == 'pepclm':
        print('Calculating PeptideCLM representations...')
        calculate_pepclm(dataset)
    elif rep == 'pepland':
        print('Calculating Pepland representations...')
        calculate_pepland(dataset)
    elif rep == 'pepfunn':
        print('Calculating pepfunn fingerprint...')
        calculate_pepfunnfp(dataset)
    elif 'esm' in rep or 'prot' in rep:
        print(f'Calculating {rep.upper()} representations...')
        calculate_esm(dataset, rep)


if __name__ == '__main__':
    typer.run(main)
