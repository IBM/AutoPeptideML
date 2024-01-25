#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
from pathlib import Path


this_directory = Path(__file__).parent
readme = (this_directory / "README.md").read_text()

requirements = [ 
    'charset-normalizer',
    'gdown',
    'graph-part',
    'networkx',
    'optuna',
    'pandarallel',
    'scikit-learn',
    'scikit-plot',
    'tokenizers',
    'torch',
    'transformers',
    'pandarallel',
    'lightgbm',
    'mdpdf'
]

test_requirements = requirements

setup(
    author="Raul Fernandez-Diaz",
    author_email='raulfd@ibm.com',
    python_requires='>=3.10',
    classifiers=[
    ],
    description="AutoML system for building trustworthy peptide bioactivity predictors",
    entry_points={
        'console_scripts': [
            'autopeptideml=autopeptideml.main:main',
            'autopeptdeml-predict=autopeptideml.main:predict',
            'autopeptideml-setup=autopeptideml.data.preprocess_db:main'
        ],
    },
    install_requires=requirements,
    license="MIT",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='autopeptideml',
    name='autopeptideml',
    packages=find_packages(exclude=['examples']),
    url='https://github.ibm.com/raulfd/autopeptideml',
    version='0.1.1',
    zip_safe=False,
)
