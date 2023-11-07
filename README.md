
# AutoPeptideML

AutoPeptideML is an open-source AutoML system for helping scientists without a computational backgorund to build their own custom peptide bioactivity predictiors and provide a robust and interpretable evaluation following [DOME](https://www.nature.com/articles/s41592-021-01205-4) community guidelines. For small datasets, please consider using the [AutoPeptideML webserver](http://peptide.ucd.ie/AutoPeptideML).


<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/IBM/AutoPeptideML/blob/main/doc/imgs/APML_dark.png" width="850" class="center">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/IBM/AutoPeptideML/blob/main/doc/imgs/APML_light.png" width="850" class="center">
  <img alt="Shows the logo for the AutoPeptideML package." src="https://github.com/IBM/AutoPeptideML/blob/main/doc/imgs/APML_light.png">
</picture>

## Contents

<details open><summary><b>Table of Contents</b></summary>

- [Intallation Guide](#installation)
- [Benchmark Data](#benchmark)
- [Documentation](#documentation)
- [Examples](#examples)
- [License](#license)
- [Acknowledgements](#acknowledgements)
 </details>

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


### 2. Preparing AutoPeptideML-Peptipedia Database

Download and prepare the Peptipedia Database by running:

```bash
autopeptideml-setup
```

If any error happens during the download, it may be caused by not having the latest version of the [gdown](https://github.com/wkentaro/gdown) library.
Please run the following code to get the latest version:

```bash
pip install --upgrade --no-cache-dir gdown
```

### 3. Third-party dependencies

For using MMSeqs as alignment algorithm is necessary install it in the environment:

```bash 
conda install -c bioconda mmseqs2
```

For using Needleman-Wunch:

```bash
conda install -c bioconda emboss
```

If installation not in conda environment, please check installation instructions for your particular device:

- Linux:
  ```bash
  wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
  tar xvfz mmseqs-linux-avx2.tar.gz
  export PATH=$(pwd)/mmseqs/bin/:$PATH
  ```

  ```bash
  sudo apt install emboss
  ```

  ```bash
  sudo apt install emboss
  ```

- Windows: Download binaries from [EMBOSS](https://emboss.sourceforge.net/download/) and [MMSeqs2-latest](https://mmseqs.com/latest/mmseqs-win64.zip)

- Mac:
  ```bash
  sudo port install emboss
  brew install mmseqs2
  ```

## Benchmark data <a name="benchmark"></a>

Data used to benchmark our approach has been selected from the benchmarks collected by [Du et al, 2023](https://academic.oup.com/bib/article-abstract/24/3/bbad135/7107929). A new set of benchmarks was constructed from the original set following the new data acquisition and dataset partitioning methods within AutoPeptideML. To download the datasets:

- **Original UniDL4BioPep Benchmarks:** Please check the project [Github Repository](https://github.com/dzjxzyd/UniDL4BioPep/tree/main).
- **New AutoPeptideML Benchmarks:** Can be downloaded from this [link](https://drive.google.com/u/0/uc?id=1UmDu773CdkBFqkitK550uO6zoxhU1bUB&export=download).

## Documentation <a name="documentation"></a>

<details><summary><b>1. Model builder options</summary></b><a name="builder"></a>

**Dataset construction**

- `dataset`: File with positive peptides in `FASTA` or `CSV` file. It can also contain negative peptides in which case the files should contain the labels (0: negative or 1: positive) either in the header (`FASTA`) or in column `Y` (`CSV`).

- `--balance`: If `True`, it balances the datasets by oversampling the underrepresented label.

-  `--autosearch`: If `True`, it searches for negative peptides.
-  `--autosearch_tags`: Comma separated list of tags that may overlap with positive activity that are going to be excluded from the negative peptides.
-  `--autosearch_proportion`: Negative:positive ration when automatically drawing negative controls from the bioactive peptides database (Default: 1.0).


**Output**
- `--outputdirdir`: Output directory (Default: `./apml_result/apml_result`).

**Protein Language Model**
- `--plm`: Protein Language Model for computing peptide representations. Available options: `esm2-8m`, `esm2-35m`, `esm2-150m`, `esm2-650m`, `esm2-3b`, `esm2-15b`, `esm1b`, `prot-t5-xxl`, `prot-t5-xl`, `protbert`, `prost-t5`. (Default: `esm2-8m`). Please note: Larger Models might not fit into GPU RAM, if it is necessary for your purposes, please create a new issue.
- `--plm_batch_size`: Number of peptides for which to batch the PLM computation.(Default: 12).

**Dataset Partitioning**
- `--test_partition`: Whether to divide the dataset in train/test splits. (Default: `True`).
- `--test_threshold`: Maximum sequence identity allowed between train and test. (Default: 0.3).
- `--test_size`: Proportion of data to be assigned to evaluation set. (Default: 0.2).
- `--test_alignment`: Alignment algorithm used for computing sequence identities. Available options: `mmseqs`, `mmseqs+prefilter`, `needle`. (Default: `mmseqs+prefilter`).
- `--splits`: Path to directory with train and test splits. Expected contents: `train.csv` and `test.csv`.

- `--val_partition`: Whether to divide dataset in train/validation folds.
- `--val_method`: Method to use for creating train/validation folds. Options available: `random`, `graph-part`. (Default: `random`)
- `--val_threshold`: Maximum sequence identity allowed between train and validation. (Default: 0.5).
- `--val_alignment`:  Alignment algorithm used for computing sequence identities. Available options: `mmseqs`, `mmseqs+prefilter`, `needle`. (Default: `mmseqs+prefilter`).
- `--val_n_folds`: Number of folds (Default: 10).
- `--folds`: Path to directory with train/validation folds. Expected contents: `train_{fold}.csv` and `valid_{fold}.csv`.

**Model Selection and Hyperparameter Optimisation**

- `--config`: Name of one of the pre-defined configuration files (see `autopeptideml/data/configs`) or path to a custom configuration file (see next section).

**Other**
- `--verbose`: Whether to display information about runtime (Default: True).
- `--threads`: Number of threads to use for parallelization. (Default: Number of cores in the machine).
- `--seed`: Seed for pseudorandom number generators. Controls stochastic processes. (Default: 42)
</details>

<details><summary><b>2. Predict</summary></b><a name="predict"></a>

- `dataset`: File with problem peptides in `FASTA` or `CSV` file.
- `--ensemble`: Path to the a file containing a previous AutoPeptideML result.
- `--outputdir`: Output directory (Default: `./apml_predictions`).
- `--verbose`: Whether to display information about runtime (Default: True).
- `--threads`: Number of threads to use for parallelization. (Default: Number of cores in the machine).
- `--plm`: Protein Language Model for computing peptide representations. Must be the same as used to train the model. Available options: `esm2-8m`, `esm2-35m`, `esm2-150m`, `esm2-650m`, `esm2-3b`, `esm2-15b`, `esm1b`, `prot-t5-xxl`, `prot-t5-xl`, `protbert`, `prost-t5`. (Default: `esm2-8m`). Please note: Larger Models might not fit into GPU RAM, if it is necessary for your purposes, please create a new issue.
- `--plm_batch_size`: Number of peptides for which to batch the PLM computation.(Default: 12).

</details>

<details><summary><b>3. Hyperparameter Optimisation and model selection</summary></b><a name="hpo"></a>

The experiment configuration is a file in `JSON` format describing the hyperparameter optimisation search space and the composition of the final ensemble. The first level of the file is a dictionary with a single key (`ensemble` or `model_selection` or `model_selection`) and a list of search spaces for the hyperparameter optimisation. For each model within the `ensemble` list, `n` different models will be trained one per cross-validation fold; in the case of `model_selection`, only one of the algorithms will comprise the final ensemble; in the case of `model_selection`, only one of the algorithms will comprise the final ensemble.

Each experiment requires the following fields:

- `model`: Defines the ML algorithm. Options: `KNearestNeighbours`, `SVM`, `RFC`, `XGBoost`, `LGBM`, `MLP`, and `UniDL4BioPep`. More options will be added in subsequent releases and they can be implemented upon request.
- `trials`: Defines the number of iterations for the hyperparameter optimisation search.
- `optimization_metric`: Defines the metric that should be used for directing the optimisation search. Always, the metric will be calculated as the average across the `n` cross-validation folds. For the metrics available all of the binary classification within the list in the [scikit-learn documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) are supported (Default: Matthew's correlation coefficient, MCC).
- `hyperparameter-space`: List of dictionaries that defines the hyperparameter search space proper. Each of the dictionaries within correspond to a different hyperparameter and may have the following fields:
   - `name`: It has to correspond with the corresponding hyperparameter in the model implementation. Most of the simpler ML models use the `scikit-learn` implementation, `LGBM` uses the Microsoft implementation (More information on [LGBM Repository](https://github.com/microsoft/LightGBM)) and `UniDL4BioPep` uses the PyTorch implementation (More information on [UniDL4BioPep PyTorch Repository](https://github.com/David-Dingle/UniDL4BioPep_ASL_PyTorch)), though for this model hyperparameter optimisation is not recommended.
   - `type`: Defines the type of hyperparameter. Options: `int`, `float`, or `categorical`. 
   - `min` and `max`: Defines the lower and upper bounds of the search space for types `int` and `float`.
   - `log`: Boolean value that defines whether the search should be done in logarithmic space or not. Accelerates searches through vast spaces for example for learning rates (1e-7 to 1). It is not optional.
   - `value`: Defines the list of options available for a hyperparameter of type `categorical` for example types of kernel (`linear`, `rbf`, `sigmoid`) for a Support Vector Machine.

There is an example available in the [default configuration file](https://github.ibm.com/raulfd/AutoPeptideML/blob/main/autopeptideml/data/configs/default_config.json).

</details>

<details><summary><b>4. API</summary></b><a name="api"></a>

Example notebooks and documentation in how to use the API can be found in the `examples` directory.

</details>


## Examples <a name="examples"></a>

```bash
autopeptideml dataset.csv
```

```bash
autopeptideml dataset.csv --val_method graph-part --val_threshold 0.3 --val_alignment needle
autopeptideml dataset.csv
```

```bash
autopeptideml dataset.csv --val_method graph-part --val_threshold 0.3 --val_alignment needle
```



License <a name="license"></a>
-------
AutoPeptideML is an open-source software licensed under the MIT Clause License. Check the details in the [LICENSE](https://github.com/raulfd-creator/biobrigit/blob/master/LICENSE) file.

Credits <a name="acknowledgements"></a>
-------

Special thanks to [Silvia González López](https://www.linkedin.com/in/silvia-gonz%C3%A1lez-l%C3%B3pez-717558221/) for designing the AutoPeptideML logo and to [Marcos Martínez Galindo](https://www.linkedin.com/in/marcosmartinezgalindo) for his aid in setting up the AutoPeptideML webserver.
