<div align="center">

  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://ibm.github.io/AutoPeptideML/imgs/APML_light.png" height="250x">
    <img alt="logo" src="https://ibm.github.io/AutoPeptideML/imgs/APML_dark.png">
  </picture>

  <h1>AutoPeptideML</h1>

  <p>
    <strong>AutoML system for building trustworthy peptide bioactivity predictors</strong>
  </p>

  <p>    
    
<a href="https://ibm.github.io/AutoPeptideML/"><img alt="Tutorials" src="https://img.shields.io/badge/docs-tutorials-green" /></a>
<a href="https://github.com/IBM/AutoPeptideML/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/IBM/AutoPeptideML" /></a>
<a href="https://pypi.org/project/autopeptideml/"><img src="https://img.shields.io/pypi/v/autopeptideml" /></a>
<a href="https://static.pepy.tech/project/autopeptideml/"><img src="https://static.pepy.tech/badge/autopeptideml" /></a>
<a target="_blank" href="https://colab.research.google.com/github/IBM/AutoPeptideML/blob/main/examples/AutoPeptideML_Collab.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>



  </p>
</div>

- **Documentation:**  <a href="https://ibm.github.io/AutoPeptideML/" target="_blank">https://ibm.github.io/AutoPeptideML</a>
- **Source Code:** <a href="https://github.com/IBM/AutoPeptideML" target="_blank">https://github.com/IBM/AutoPeptideML</a>
- **Webserver:** <a href="http://peptide.ucd.ie/AutoPeptideML" target="_blank">http://peptide.ucd.ie/AutoPeptideML</a>
- **Google Collaboratory Notebook:** <a href="https://colab.research.google.com/github/IBM/AutoPeptideML/blob/main/examples/AutoPeptideML_Collab.ipynb" target="_blank">AutoPeptideML_Collab.ipynb</a>
- **Blog post:** <a href="https://portal.valencelabs.com/blogs/post/autopeptideml-building-peptide-bioactivity-predictors-automatically-IZZKbJ3Un0qjo4i" target="_blank">Portal - AutoPeptideML v. 1.0 Tutorial</a>
- **Papers:** 
  - [AutoPeptideML (v. 1.0)](https://doi.org/10.1093/bioinformatics/btae555)
  - [ML Generalization from canonical to non-canonical peptides](https://doi.org/10.26434/chemrxiv-2025-ggp8n)

AutoPeptideML allows researchers without prior knowledge of machine learning to build models that are:

- **Trustworthy:** Robust evaluation following community guidelines for ML evaluation reporting in life sciences [DOME](https://www.nature.com/articles/s41592-021-01205-4).
- **Interpretable:** Output contains a PDF summary of the model evaluation explaining how to interpret the results to understand how reliable the model is.
- **Reproducible:** Output contains all necessary information for other researchers to reproduce the training and verify the results.
- **State-of-the-art:** Models generated with this system are competitive with state-of-the-art handcrafted approaches.

To use version 1.0, which may be necessary for retrocompatibility with previously built models, please defer to the branch: [AutoPeptideML v.1.0.6](https://github.com/IBM/AutoPeptideML/tree/apml-1.0.6)

## Contents

<details open markdown="1"><summary><b>Table of Contents</b></summary>

- [Model builder](#helper)
- [Prediction](#prediction)
- [Benchmark Data](#benchmark)
- [Intallation Guide](#installation)
- [Documentation](#documentation)
- [License](#license)
- [Acknowledgements](#acknowledgements)
 </details>


## Model builder <a name="helper"></a>

In order to build a new model, AutoPeptideML (v.2.0), introduces a new utility to automatically prepare an experiment configuration file, to i) improve the reproducibility of the pipeline and ii) to keep a user-friendly interface despite the much increased flexibility.

```bash
autopeptideml prepare-config
```
This launches an interactive CLI that walks you through:

- Choosing a modeling task (classification or regression)
- Selecting input modality (macromolecules or sequences)
- Loading and parsing datasets (csv, tsv, or fasta)
- Defining evaluation strategy
- Picking models and representations
- Setting hyperparameter search strategy and training parameters


You’ll be prompted to answer various questions like:

```
- What is the modelling problem you're facing? (Classification or Regression)

- How do you want to define your peptides? (Macromolecules or Sequences)

- What models would you like to consider? (knn, adaboost, rf, etc.)
```

And so on. The final config is written to:

```
<outputdir>/config.yml
```

This config file allows for easy reproducibility of the results, so that anyone can repeat the training processes. You can check the configuration file and make any changes you deem necessary. Finally, you can build the model by simply running:

```
autopeptideml build-model --config-path <outputdir>/config.yml
```

## Prediction <a name="prediction"></a>

In order to use a model that has already built you can run:

```bash
autopeptideml predict <model_outputdir> <features_path> <feature_field> --output-path <my_predictions_path.csv>
```

Where `<features_path>` is the path to a `CSV` file with a column `features_field` that contains the peptide sequences/SMILES. The output file `<my_predictions_path>` will contain the original data with two additional columns `score` (which are the predictions) and `std` which is the standard deviation between the predictions of the models in the ensemble, which can be used as a measure of the uncertainty of the prediction.

## Benchmark data <a name="benchmark"></a>

Data used to benchmark our approach has been selected from the benchmarks collected by [Du et al, 2023](https://academic.oup.com/bib/article-abstract/24/3/bbad135/7107929). A new set of benchmarks was constructed from the original set following the new data acquisition and dataset partitioning methods within AutoPeptideML. To download the datasets:

- **Original UniDL4BioPep Benchmarks:** Please check the project [Github Repository](https://github.com/dzjxzyd/UniDL4BioPep/tree/main).
- **⚠️ New AutoPeptideML Benchmarks (Amended version):** Can be downloaded from this [link](https://drive.google.com/u/0/uc?id=1UmDu773CdkBFqkitK550uO6zoxhU1bUB&export=download). Please note that these are not exactly the same benchmarks as used in the paper (see [Issue #24](https://github.com/IBM/AutoPeptideML/issues/24) for more details).
- **PeptideGeneralizationBenchmarks:** Benchmarks evaluating how peptide representation methods generalize from canonical (peptides composed of the 20 standard amino acids) to non-canonical (peptides with non-standard amino acids or other chemical modifications). Check out the [paper pre-print](https://chemrxiv.org/engage/chemrxiv/article-details/67d2f3ae81d2151a023d64f8). They have their own dedicated repository: [PeptideGeneralizationBenchmarks Github repository](https://github.com/IBM/PeptideGeneralizationBenchmarks).

## Installation <a name="installation"></a>

Installing in a conda environment is recommended. For creating the environment, please run:

```bash
conda create -n autopeptideml python
conda activate autopeptideml
```

### 1. Python Package

#### 1.1.From PyPI


```bash
pip install autopeptideml
```

#### 1.2. Directly from source

```bash
pip install git+https://github.com/IBM/AutoPeptideML
```

### 2. Third-party dependencies

To use MMSeqs2 [https://github.com/steineggerlab/mmseqs2](https://github.com/steineggerlab/mmseqs2)

  ```bash
  # static build with AVX2 (fastest) (check using: cat /proc/cpuinfo | grep avx2)
  wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz; tar xvfz mmseqs-linux-avx2.tar.gz; export PATH=$(pwd)/mmseqs/bin/:$PATH

  # static build with SSE4.1  (check using: cat /proc/cpuinfo | grep sse4)
  wget https://mmseqs.com/latest/mmseqs-linux-sse41.tar.gz; tar xvfz mmseqs-linux-sse41.tar.gz; export PATH=$(pwd)/mmseqs/bin/:$PATH

  # static build with SSE2 (slowest, for very old systems)  (check using: cat /proc/cpuinfo | grep sse2)
  wget https://mmseqs.com/latest/mmseqs-linux-sse2.tar.gz; tar xvfz mmseqs-linux-sse2.tar.gz; export PATH=$(pwd)/mmseqs/bin/:$PATH

  # MacOS
  brew install mmseqs2  
  ```

  To use Needleman-Wunch, either:

  ```bash
  conda install -c bioconda emboss
  ```
  or

  ```bash
  sudo apt install emboss
  ```

To use ECFP fingerprints:

```bash
pip install rdkit
```

To use MAPc fingeprints:

```bash
pip install mapchiral
```

To use PepFuNN fingeprints:

```bash
pip install git+https://github.com/novonordisk-research/pepfunn
```

To use PeptideCLM:

```bash
pip install smilesPE
```

## Documentation <a name="documentation"></a>

### Configuration file

#### Top-level structure

```yaml
pipeline: {...}
databases: {...}
test: {...}
val: {...}
train: {...}
representation: {...}
outputdir: "path/to/experiment_results"
```

#### `pipeline`
Defines the preprocessing pipeline depending on the modality (`mol` or `seqs`). It includes data cleaning and transformations, such as:

- `filter-smiles`
- `canonical-cleaner`
- `sequence-to-smiles`
- `smiles-to-sequences`

The name of a pipeline object has to include the word `pipe`. Pipelines can be elements within a pipeline. Here, is an example. Aggregate will combine the output from the different elements. In this case, the two elements process SMILES and sequences independently and then combine them into a single datastream.


```yaml
pipeline:
  name: "macromolecules_pipe"
  aggregate: true
  verbose: false
  elements:
    - pipe-smiles-input: {...}
    - pipe-seq-input: {...}

```

### `databases`

Defines dataset paths and how to interpret them.

**Required:**
- `path`: Path to main dataset.
- `feat_fields`: Column name with SMILES or sequences.
- `label_field`: Column with classification/regression labels.
- `verbose`: Logging flag.

**Optional:**
- `neg_database`: If using negative sampling.
- `path`: Path to negative dataset.
- `feat_fields`: Feature column.
- `columns_to_exclude`: Bioactivity columns to ignore.

```yaml
databases:
  dataset:
    path: "data/main.csv"
    feat_fields: "sequence"
    label_field: "activity"
    verbose: false
  neg_database:
    path: "data/negatives.csv"
    feat_fields: "sequence"
    columns_to_exclude: ["to_exclude"]
    verbose: false
```

### `test`

Defines evaluation and similarity filtering settings.

- min_threshold: Identity threshold for filtering.
- sim_arguments: Similarity computation details.

For sequences:

- `alignment_algorithm`: `mmseqs`, `mmseqs+prefilter`, `needle`
- `denominator`: How identity is normalized: `longest`, `shortest`, `n_aligned`
- `prefilter`: Whether to use a prefilter.
- `field_name`: Name of column with the peptide sequences/SMILES
- `verbose`: Logging flag.

For molecules:

- `sim_function`: e.g., tanimoto, jaccard
- `radius`: Radius to define the substructures when computing the fingerprint
- `bits`: Size of the fingerprint, greater gives more resolution but demands more computational resources.
- `partitions`: `min`, `all`, `<threshold>`
- `algorithm`: `ccpart`, `ccpart_random`, `graph_part`
- `threshold_step`: Step size for threshold evaluation.
- `filter`: Minimum proportion of data in the test set that is acceptable (test set proportion = 20%, `filter=0.185`, does not consider test sets with less than 18.5%)
- `verbose`: Logging level.

Example:

```yaml
test:
  min_threshold: 0.1
  sim_arguments:
    data_type: "sequence"
    alignment_algorithm: "mmseqs"
    denominator: "shortest"
    prefilter: true
    min_threshold: 0.1
    field_name: "sequence"
    verbose: 2
  partitions: "all"
  algorithm: "ccpart"
  threshold_step: 0.1
  filter: 0.185
  verbose: 2
```

### `val`

Cross-validation strategy:

- `type`: `kfold` or `single`
- `k`: Number of folds.
- `random_state`: Seed for reproducibility.

### `train`
Training configuration.

Required:

- `task`: class or reg
- `optim_strategy`: Optimization strategy.
- `trainer`: grid or optuna
- `n_steps`: Number of trials (Optuna only).
- `direction`: maximize or minimize
- `metric`: mcc or mse
- `partition`: Partitioning type.
- `n_jobs`: Parallel jobs.
- `patience`: Early stopping patience.
- `hspace`: Search space.
- `representations`: List of representations to try.
- `models`:
- `type`: select or ensemble
- `elements`: model names and their hyperparameter space.

Example: 

```yaml
train:
  task: "class"
  optim_strategy:
    trainer: "optuna"
    n_steps: 100
    direction: "maximize"
    task: "class"
    metric: "mcc"
    partition: "random"
    n_jobs: 8
    patience: 20
  hspace:
    representations: ["chemberta-2", "ecfp-4"]
    models:
      type: "select"
      elements:
        knn:
          n_neighbors:
            type: int
            min: 1
            max: 20
            log: false
          weights:
            type: categorical
            values: ["uniform", "distance"]
```


### `representation`
Specifies molecular or sequence representations.

Each element includes:

- `engine`: `lm` (language model) or `fp` (fingerprint)
- `model`: Model name (e.g., chemberta-2, esm2-150m)
- `device`: `cpu`, `gpu`, or `mps`
- `batch_size`: Size per batch
- `average_pooling`: Whether to average token representations (only for `lm`)

```yaml
representation:
  verbose: true
  elements:
    - chemberta-2:
        engine: "lm"
        model: "chemberta-2"
        device: "gpu"
        batch_size: 32
        average_pooling: true
    - ecfp-4:
        engine: "fp"
        fp: "ecfp"
        radius: 2
        nbits: 2048
```

### More details about API

Please check the [Code reference documentation](https://ibm.github.io/AutoPeptideML/autopeptideml/)



License <a name="license"></a>
-------
AutoPeptideML is an open-source software licensed under the MIT Clause License. Check the details in the [LICENSE](https://github.com/IBM/AutoPeptideML/blob/master/LICENSE) file.

Credits <a name="acknowledgements"></a>
-------

Special thanks to [Silvia González López](https://www.linkedin.com/in/silvia-gonz%C3%A1lez-l%C3%B3pez-717558221/) for designing the AutoPeptideML logo and to [Marcos Martínez Galindo](https://www.linkedin.com/in/marcosmartinezgalindo) for his aid in setting up the AutoPeptideML webserver.
