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
autopeptideml prepare-config --config-path <config-path>
```
This launches an interactive CLI that walks you through:

- Choosing a modeling task (classification or regression)
- Loading and parsing datasets (csv, tsv, or fasta)
- Picking models and representations
- Automatically sampling negatives


You’ll be prompted to answer various questions like:

```
- What is the modelling problem you're facing? (Classification or Regression)

- How do you want to define your peptides? (Macromolecules or Sequences)

- What models would you like to consider? (knn, adaboost, rf, etc.)
```

And so on. The final config is written to:

```
<config-path>.yml
```

This config file allows for easy reproducibility of the results, so that anyone can repeat the training processes. You can check the configuration file and make any changes you deem necessary. Finally, you can build the model by simply running:

```
autopeptideml build-model --outdir <outdir> --config-path <outputdir>/config.yml
```

## Prediction <a name="prediction"></a>

In order to use a model that has already built you can run:

```bash
autopeptideml predict <model_outputdir> <features_path> <feature_field> --output-path <my_predictions_path.csv>
```

Where `<features_path>` is the path to a `CSV` file with a column `<features_field>` that contains the peptide sequences/SMILES. The output file `<my_predictions_path>` will contain the original data with two additional columns `score` (which are the predictions) and `std` which is the standard deviation between the predictions of the models in the ensemble, which can be used as a measure of the uncertainty of the prediction.

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


```yaml
datasets:
  main:
    feat-fields: # Column with peptide sequence/SMILES
    label-field: # Column with labels/ "Assume all entries are positives"
    path: # Path to dataset
  neg-db:
    activities-to-exclude: # List of activities to exclude
      - activity-1
      - activity-2
      ...
    feat-fields: null # Column with peptide sequence/SMILES (only if using custom database)
    path: # Path to custom database or choose: canonical, non-canonical, both
device: # Device for computing representations. Choose: cpu, mps, cuda
direction: # Direction of optimization. Choose: maximize or minimize
metric: # Metric for optimization. mse, mae require direction minimize
models: # List of machine learning algorithms to explore. List:
        # knn, svm, rf, gradboost, xgboost, lightgbm
  - model-1
  - model-2
  ...
n-trials: # Number of optimization steps. Recommended 100-200
pipeline: to-smiles # Pipeline for preprocessing. Choose: to-smiles, to-sequences
reps: # List of peptide representations to explore. List:
      # ecfp, chemberta-2, molformer-xl, peptide-clm, esm2-8m, ...
  - rep-1
  - rep-2
  ...

split-strategy: min # Strategy for splitting train/test. Choose: min, random. 
task: class # Machine learning type of problem. Choose: class or reg.
n-jobs: # Number of processes to launch. -1 uses all possible CPU cores.
```

### More details about API

Please check the [Code reference documentation](https://ibm.github.io/AutoPeptideML/autopeptideml/)



License <a name="license"></a>
-------
AutoPeptideML is an open-source software licensed under the MIT Clause License. Check the details in the [LICENSE](https://github.com/IBM/AutoPeptideML/blob/master/LICENSE) file.

Credits <a name="acknowledgements"></a>
-------

Special thanks to [Silvia González López](https://www.linkedin.com/in/silvia-gonz%C3%A1lez-l%C3%B3pez-717558221/) for designing the AutoPeptideML logo and to [Marcos Martínez Galindo](https://www.linkedin.com/in/marcosmartinezgalindo) for his aid in setting up the AutoPeptideML webserver.
