# `RepEngineLM` — Language Model Representation Engine

**Module:** `autopeptideml.reps.lms`  
**Inherits from:** [`RepEngineBase`](repenginebase.md)

## Overview

`RepEngineLM` generates dense vector embeddings for peptide sequences or SMILES strings using pre-trained transformer language models loaded from HuggingFace. It supports protein language models (ESM-2, ProtT5, ANKH, …) and small-molecule language models (MoLFormer, ChemBERTa, PeptideCLM).

**Requires:** `pip install torch transformers`

---

## Attributes

| Attribute | Type | Description |
|---|---|---|
| `engine` | `str` | Fixed to `'lm'`. |
| `device` | `str` | Compute device: `'cuda'`, `'mps'`, or `'cpu'` (auto-detected). |
| `model` | object | Loaded HuggingFace model. |
| `tokenizer` | object | Associated tokenizer. |
| `model_name` | `str` | Canonical HuggingFace model name. |
| `dimension` | `int` | Embedding dimensionality. |
| `lab` | `str` | HuggingFace organisation name (e.g. `'facebook'`, `'Rostlab'`). |
| `name` | `str` | Engine identifier in the form `lm-<model>`. |
| `average_pooling` | `bool` | If `True` (default), residue embeddings are mean-pooled per sequence. |
| `cls_token` | `bool` | If `True`, use only the `[CLS]` token embedding (takes precedence over pooling). |
| `fp16` | `bool` | If `True`, use bfloat16 precision via `torch.autocast` where supported. |

---

## Constructor

```python
RepEngineLM(
    model: str,
    average_pooling: Optional[bool] = True,
    cls_token: Optional[bool] = False,
    fp16: bool = True
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | — | Model name or short synonym. See [Available Models](#available-models). |
| `average_pooling` | `bool` | `True` | Average all token embeddings to produce a fixed-size sequence vector. |
| `cls_token` | `bool` | `False` | Use only the `[CLS]` token embedding. Overrides `average_pooling`. |
| `fp16` | `bool` | `True` | Enable bfloat16 autocast when the device supports it. |

---

## Public Methods

### `compute_reps` *(inherited)*

```python
compute_reps(
    mols: List[str],
    verbose: bool = False,
    batch_size: int = 12
) -> Union[np.ndarray, List[np.ndarray]]
```

Compute embeddings for a list of sequences/SMILES. With `average_pooling=True` returns a `(n_mols, dimension)` array.

---

### `dim`

```python
dim() -> int
```

Returns the embedding dimension of the model.

---

### `max_len`

```python
max_len() -> int
```

Returns the maximum accepted sequence length for the loaded model:

| Lab | Max length |
|---|---|
| `facebook` (ESM) | 1022 |
| `EvolutionaryScale` / `InstaDeepAI` | 2046 |
| `DeepChem` (ChemBERTa) | 512 |
| All others | 2046 |

---

### `move_to_device`

```python
move_to_device(device: str)
```

Moves the model to the specified device. Useful when the device is not available at construction time.

| Parameter | Type | Description |
|---|---|---|
| `device` | `str` | Target device: `'cpu'`, `'cuda'`, or `'mps'`. |

---

### `get_num_params`

```python
get_num_params(human_readable: bool = False) -> Union[int, str]
```

Returns the total number of trainable model parameters.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `human_readable` | `bool` | `False` | If `True`, returns a formatted string like `"650.000M"` instead of an integer. |

---

## Available Models

The `model` argument accepts either a canonical HuggingFace name or a short synonym.

| Short Synonym | Canonical Name | Dimension | Type |
|---|---|---|---|
| `esm2-8m` | `esm2_t6_8M_UR50D` | 320 | Protein LM |
| `esm2-35m` | `esm2_t12_35M_UR50D` | 480 | Protein LM |
| `esm2-150m` | `esm2_t30_150M_UR50D` | 640 | Protein LM |
| `esm2-650m` | `esm2_t33_650M_UR50D` | 1280 | Protein LM |
| `esm1b` | `esm1b_t33_650M_UR50S` | 1280 | Protein LM |
| `esm2-3b` | `esm2_t36_3B_UR50D` | 2560 | Protein LM |
| `esm2-15b` | `esm2_t48_15B_UR50D` | 5120 | Protein LM |
| `esmc-300m` | `ESMplusplus_small` | 960 | Protein LM |
| `esmc-600m` | `ESMplusplus_large` | 1152 | Protein LM |
| `prot-t5-xl` | `prot_t5_xl_half_uniref50-enc` | 1024 | Protein LM |
| `prot-t5-xxl` | `prot_t5_xxl_uniref50` | 1024 | Protein LM |
| `protbert` | `prot_bert` | 1024 | Protein LM |
| `prost-t5` | `ProstT5` | 1024 | Protein LM |
| `ankh-base` | `ankh-base` | 768 | Protein LM |
| `ankh-large` | `ankh-large` | 1536 | Protein LM |
| `molformer-xl` | `MoLFormer-XL-both-10pct` | 768 | Small molecule LM |
| `chemberta-2` | `ChemBERTa-77M-MLM` | 384 | Small molecule LM |
| `chemberta-3` | `ChemBERTa-100M-MLM` | 768 | Small molecule LM |
| `peptideclm` | `PeptideCLM-23M-all` | 768 | Peptide SMILES LM |
| `peptidemtr` | `PeptideMTR_lg` | 1024 | Peptide LM |
| `nt2-500m-ms` | `nucleotide-transformer-v2-500m-multi-species` | 1024 | Nucleotide LM |

!!! note "PeptideCLM"
    Using `peptideclm` requires the `smilesPE` package: `pip install smilesPE`.
    The tokenizer vocabulary files are downloaded automatically on first use from the PeptideCLM GitHub repository.

!!! warning "MoLFormer"
    MoLFormer does not support `transformers >= 5.0.0`. Pin to `transformers==4.41.2` if you encounter issues.

---

## Example

```python
from autopeptideml.reps.lms import RepEngineLM

# Load ESM-2 8M on CPU
engine = RepEngineLM(model='esm2-8m', average_pooling=True)

sequences = ['ACDEFGHIKLMNPQRSTVWY', 'AACGWYLP']
embeddings = engine.compute_reps(sequences, verbose=True, batch_size=4)

print(embeddings.shape)   # (2, 320)
print(engine.dim())       # 320
print(engine.get_num_params(human_readable=True))  # e.g. "8.000M"
```

---

## Preprocessing per Model Family

`_preprocess_batch` performs model-specific sequence preparation:

| Lab / Model | Preprocessing |
|---|---|
| `Rostlab` (ProtT5, ProtBERT) | Space-delimited residues: `"A C D E F …"` |
| `ProstT5` | Prepended with `"<AA2fold> "` |
| All others | Sequences truncated to `max_len()` only |
