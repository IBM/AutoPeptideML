# `RepEngineSkfp` — scikit-fingerprints Engine

**Module:** `autopeptideml.reps.fps`  
**Inherits from:** [`RepEngineBase`](repenginebase.md)

## Overview

`RepEngineSkfp` wraps any [scikit-fingerprints](https://scikit-fingerprints.github.io/scikit-fingerprints/) (`skfp`) fingerprint class as a drop-in `RepEngineBase`-compatible engine. It gives access to 28 fingerprint families beyond the RDKit-backed ones available in [`RepEngineFP`](repenginefp.md), including MACCS, AtomPair, TopologicalTorsion, Avalon, PubChem, Mordred, and more.

**Requires:** `pip install scikit-fingerprints`  
(RDKit is also required as a transitive dependency.)

---

## Attributes

| Attribute | Type | Description |
|---|---|---|
| `engine` | `str` | Fixed to `'skfp'`. |
| `name` | `str` | Auto-generated as `'skfp-<rep>'`, e.g. `'skfp-maccs'`. |
| `generator` | `BaseFingerprintTransformer` | The underlying skfp transformer instance. |

---

## Constructor

```python
RepEngineSkfp(rep: str, **kwargs)
```

| Parameter | Type | Description |
|---|---|---|
| `rep` | `str` | Lowercase fingerprint key (see [Supported fingerprints](#supported-fingerprints)). |
| `**kwargs` | `Any` | Forwarded verbatim to the skfp fingerprint constructor (e.g. `fp_size`, `radius`, `count`). |

---

## Methods

### `compute_reps` *(inherited)*

```python
compute_reps(
    mols: List[str],
    verbose: bool = False,
    batch_size: int = 12
) -> np.ndarray
```

Compute fingerprints for a list of SMILES strings. Returns an array of shape `(n_mols, dim)`.

---

### `dim`

```python
dim() -> int
```

Returns `generator.n_features_out` — the feature dimensionality reported by the underlying skfp transformer.

---

### `_preprocess_batch`

```python
_preprocess_batch(batch: List[str]) -> List[str]
```

Returns the batch unchanged. `skfp` transformers accept SMILES strings directly and handle `Mol` conversion internally.

---

### `_rep_batch`

```python
_rep_batch(batch: List[str]) -> np.ndarray
```

Delegates to `generator.transform(batch)`. Returns a dense `np.ndarray` of shape `(len(batch), dim)`.

---

### `_load_generator`

```python
_load_generator(rep: str, **kwargs) -> BaseFingerprintTransformer
```

Looks up `rep` in the internal class map and instantiates the matching skfp class with `**kwargs`. Raises `NotImplementedError` for unknown keys.

---

## Supported fingerprints

| Key | skfp class | Fixed `dim` | Notes |
|---|---|---|---|
| `atompair` | `AtomPairFingerprint` | `fp_size` | Hashed atom-pair counts |
| `autocorr` | `AutocorrFingerprint` | 192 | 2D autocorrelation descriptors |
| `avalon` | `AvalonFingerprint` | `fp_size` | Avalon substructure fingerprint |
| `bcut2d` | `BCUT2DFingerprint` | 64 | BCUT2D descriptors |
| `ecfp` | `ECFPFingerprint` | `fp_size` | Extended connectivity (Morgan); pass `use_pharmacophoric_invariants=True` for FCFP |
| `erg` | `ERGFingerprint` | 315 | Extended reduced graph |
| `estate` | `EStateFingerprint` | 79 | Electrotopological state |
| `functionalgroups` | `FunctionalGroupsFingerprint` | 85 | Functional group presence |
| `ghosecrippen` | `GhoseCrippenFingerprint` | 110 | Ghose-Crippen atom types |
| `klekotaroth` | `KlekotaRothFingerprint` | `fp_size` | Klekota-Roth substructure |
| `laggner` | `LaggnerFingerprint` | 307 | Laggner substructure |
| `layered` | `LayeredFingerprint` | `fp_size` | RDKit layered fingerprint |
| `lingo` | `LingoFingerprint` | `fp_size` | SMILES n-gram similarity |
| `maccs` | `MACCSFingerprint` | 166 | MACCS structural keys |
| `map` | `MAPFingerprint` | `fp_size` | MinHashed atom-pair |
| `mhfp` | `MHFPFingerprint` | `fp_size` | MinHashed fingerprint |
| `mordred` | `MordredFingerprint` | 1613 | Mordred 2D descriptors |
| `mqns` | `MQNsFingerprint` | 42 | Molecular quantum numbers |
| `pattern` | `PatternFingerprint` | `fp_size` | RDKit pattern fingerprint |
| `pharmacophore` | `PharmacophoreFingerprint` | `fp_size` | 2D pharmacophore |
| `pubchem` | `PubChemFingerprint` | 881 | PubChem substructure keys |
| `rdkit` | `RDKitFingerprint` | `fp_size` | RDKit path fingerprint |
| `rdkit2d` | `RDKit2DDescriptorsFingerprint` | 200 | RDKit 2D descriptors |
| `secfp` | `SECFPFingerprint` | `fp_size` | SMILES extended connectivity |
| `topologicaltorsion` | `TopologicalTorsionFingerprint` | `fp_size` | Topological torsion |
| `usr` | `USRFingerprint` | 12 | Ultrafast shape recognition (3D) |
| `usrcat` | `USRCATFingerprint` | 60 | USR + CREDO atom types (3D) |
| `vsa` | `VSAFingerprint` | 71 | Van der Waals surface area bins |

> **3D fingerprints** (`usr`, `usrcat`) require molecules with pre-computed conformations. Pass RDKit `Mol` objects with the `conf_id` property set rather than bare SMILES strings.

---

## Examples

### MACCS keys (fixed 166-bit)

```python
from autopeptideml.reps.fps import RepEngineSkfp

engine = RepEngineSkfp('maccs')
smiles = [
    'C[C@H](N)C(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)O',  # Ala-Arg-Gly
    'N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CS)C(=O)O',          # Phe-Cys
]
X = engine.compute_reps(smiles)
print(X.shape)    # (2, 166)
print(engine.dim())  # 166
```

### ECFP via skfp (variable bit size, count variant)

```python
engine = RepEngineSkfp('ecfp', fp_size=2048, radius=3, count=True)
X = engine.compute_reps(smiles)
print(X.shape)    # (2, 2048)
print(engine.dim())  # 2048
```

### AtomPair fingerprint

```python
engine = RepEngineSkfp('atompair', fp_size=512)
X = engine.compute_reps(smiles)
print(X.shape)    # (2, 512)
```

---

## Notes

- `RepEngineSkfp` and [`RepEngineFP`](repenginefp.md) both live in `autopeptideml.reps.fps` and share the same `RepEngineBase` interface.
- For ECFP / FCFP via RDKit (the existing path in `build_models`), continue to use `RepEngineFP`. `RepEngineSkfp('ecfp', ...)` is an independent implementation backed by scikit-fingerprints.
- The class map is populated lazily: importing `autopeptideml.reps.fps` does **not** require `scikit-fingerprints` to be installed until `RepEngineSkfp` is actually instantiated.
