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

- **Main branch:** <a href="https://github.com/IBM/AutoPeptideML" target="_blank">https://github.com/IBM/AutoPeptideML</a>
- **Webserver:** <a href="http://peptide.ucd.ie/AutoPeptideML" target="_blank">http://peptide.ucd.ie/AutoPeptideML</a>
- **Paper:** <a href="https://doi.org/10.1093/bioinformatics/btae555" target="_blank">https://doi.org/10.1093/bioinformatics/btae555</a>
- **Models and Paper SI**: <a href="https://zenodo.org/records/14591403/files/AutoPeptideML_SI.tar.gz?download=1" target="_blank">Zenodo Repository</a>

## Introduction

This branch of the AutoPeptideML project contains the code and data to reproduce the results from the [preprint]() "How to generalise machine learning models to both canonical and non-canonical peptides", where we studied the ability of peptide representation techniques to produce models able to generalise between canonical and non-canonical peptides  See [main branch](https://github.com/IBM/AutoPeptideML) for more details about the software.

## Contents

<details open markdown="1"><summary><b>Table of Contents</b></summary>

- [Intallation Guide](#installation)
- [Benchmark Data](#benchmark)
- [License](#license)
- [Acknowledgements](#acknowledgements)
 </details>

## Installation <a name="installation"></a>

Installing in a conda environment is recommended. For creating the environment, please run:

```bash
conda create -n joint_model python
conda activate joint_model
```

### 1. Python Package

#### 1.1.From PyPI


```bash
pip install autopeptideml
pip install rdkit mapchiral SmilesPE sentencepiece
```

#### 1.2. Directly from source

```bash
pip install git+https://github.com/IBM/AutoPeptideML
```

### 2. Third-party dependencies

Check main [Hestia-GOOD repo](https://github.com/IBM/Hestia-GOOD) for more details regarding specific installation of third-party tools for similarity calculation.

## Benchmark data <a name="benchmark"></a>

The benchmark data used is located in the [downstream_data]() folder. The partitions selected for each dataset are located in [partitiosn]() folder.


## 2. Get representations

```bash
python code/represent_peptides.py dataset representation
```

## 3. Run evaluation for similarity functions

```bash
python code/evaluation.py dataset svm sim_function fp[na if mmseqs representation radius[0 if mmseqs]
```

## 4. Run evaluation for representations or combinations or representations

```bash
python code/evaluation_reps.py dataset downstream_algorithm representation1,representation2 0 0
```

## 5. Run evaluation for representation with canonical and non-canonical test sets

Here dataset, cannot be preceded by `c-` or `nc-`. It would be `binding` rather than `c-binding` as in previous steps.

```bash
python code/evaluation_joint.py dataset downstream_algorithm representation 0 0
```

License <a name="license"></a>
-------
AutoPeptideML is an open-source software licensed under the MIT Clause License. Check the details in the [LICENSE](https://github.com/IBM/AutoPeptideML/blob/master/LICENSE) file.

Credits <a name="acknowledgements"></a>
-------

Special thanks to [Silvia González López](https://www.linkedin.com/in/silvia-gonz%C3%A1lez-l%C3%B3pez-717558221/) for designing the AutoPeptideML logo and to [Marcos Martínez Galindo](https://www.linkedin.com/in/marcosmartinezgalindo) for his aid in setting up the AutoPeptideML webserver.
