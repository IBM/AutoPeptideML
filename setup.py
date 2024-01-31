#!/usr/bin/env python

"""The setup script."""
import os
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
    'scipy<=1.11.4',
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
files = [f'autopeptideml/data/peptipedia/{file}' for file in
         os.listdir('autopeptideml/data/peptipedia')]
files.append('autopeptideml/data/bioactivities.txt')
files.append('autopeptideml/data/readme_ex.md')
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
    data_files=[('', files)],
    include_package_data=True,
    keywords='autopeptideml',
    name='autopeptideml',
    packages=find_packages(exclude=['examples']),
    url='https://github.ibm.com/raulfd/autopeptideml',
    version='0.2.8',
    zip_safe=False,
)
