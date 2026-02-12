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
<a target="_blank" href="https://colab.research.google.com/github/IBM/AutoPeptideML/blob/main/examples/AutoPeptideML2.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

  </p>
</div>

- **Documentation:** <a href="https://ibm.github.io/AutoPeptideML/" target="_blank">https://ibm.github.io/AutoPeptideML</a>
- **Source Code:** <a href="https://github.com/IBM/AutoPeptideML" target="_blank">https://github.com/IBM/AutoPeptideML</a>
- **Webserver:** <a href="http://peptide.ucd.ie/autopeptideml" target="_blank">http://peptide.ucd.ie/autopeptideml</a>
- **Google Collaboratory Notebook:** <a href="https://colab.research.google.com/github/IBM/AutoPeptideML/blob/main/examples/AutoPeptideML2.ipynb" target="_blank">AutoPeptideML_Collab.ipynb</a>
- **Blog post:** <a href="https://portal.valencelabs.com/blogs/post/autopeptideml-building-peptide-bioactivity-predictors-automatically-IZZKbJ3Un0qjo4i" target="_blank">Portal - AutoPeptideML v. 1.0 Tutorial</a>
- **Papers:**
  - [AutoPeptideML (v. 1.0)](https://doi.org/10.1093/bioinformatics/btae555)
  - [ML Generalization from standard to modified peptides](https://link.springer.com/article/10.1186/s13321-025-01115-z)

AutoPeptideML allows researchers without prior knowledge of machine learning to build models that are:

- **Trustworthy:** Robust evaluation following community guidelines for ML evaluation reporting in life sciences [DOME](https://www.nature.com/articles/s41592-021-01205-4).
- **Interpretable:** Output contains a PDF summary of the model evaluation explaining how to interpret the results to understand how reliable the model is.
- **Reproducible:** Output contains all necessary information for other researchers to reproduce the training and verify the results.
- **State-of-the-art:** Models generated with this system are competitive with state-of-the-art handcrafted approaches.

To use version 1.0, which may be necessary for retrocompatibility with previously built models, please defer to the branch: [AutoPeptideML v.1](https://github.com/IBM/AutoPeptideML/tree/apml-1)

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

In order to build a new model, AutoPeptideML (v.2.0) guides you through the process through a series of prompts.

```bash
autopeptideml build-model
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
<outputdir>/setup-config.yml
```

This config file allows for easy reproducibility of the results, so that anyone can repeat the training processes. You can check the configuration file and make any changes you deem necessary. Finally, you can build the model by simply running:

```
autopeptideml build-model --outdir <outdir> --config-path <outputdir>/setup-config.yml
```

## Prediction <a name="prediction"></a>

In order to use a model that has already built you can run:

```bash
autopeptideml predict <result_dir> <features_path> --feature-field <feature_field> --output-path <my_predictions_path.csv>
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
conda install quarto -c conda-forge
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

### More details about API

Please check the [Code reference documentation](https://ibm.github.io/AutoPeptideML/autopeptideml/)

## License <a name="license"></a>

AutoPeptideML is an open-source software licensed under the MIT Clause License. Check the details in the [LICENSE](https://github.com/IBM/AutoPeptideML/blob/master/LICENSE) file.

## Credits <a name="acknowledgements"></a>

Special thanks to [Silvia González López](https://www.linkedin.com/in/silvia-gonz%C3%A1lez-l%C3%B3pez-717558221/) for designing the AutoPeptideML logo and to [Marcos Martínez Galindo](https://www.linkedin.com/in/marcosmartinezgalindo) for his aid in setting up the AutoPeptideML webserver.
