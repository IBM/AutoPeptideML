#!/usr/bin/env python

"""The setup script."""
import os
from setuptools import setup, find_packages
from pathlib import Path


this_directory = Path(__file__).parent
readme = (this_directory / "README.md").read_text()

requirements = [
    'optuna',
    'scikit-learn',
    'typer',
    'mljar-scikit-plot',
    'tokenizers',
    'torch',
    'transformers',
    'lightgbm',
    'xgboost',
    'mdpdf',
    'onnxmltools',
    'skl2onnx',
    'onnxruntime',
    'hestia-good'
]

test_requirements = requirements
files = [f'autopeptideml/data/peptipedia/{file}' for file in
         os.listdir('autopeptideml/data/peptipedia')]
files.append('autopeptideml/data/bioactivities.txt')
files.append('autopeptideml/data/readme_ex.md')
setup(
    author="Raul Fernandez-Diaz",
    author_email='raulfd@ibm.com',
    python_requires='>=3.9',
    classifiers=[
    ],
    description="AutoML system for building trustworthy peptide bioactivity predictors",
    entry_points={
        'console_scripts': [
            'apml=autopeptideml.main:_build_model',
            'apml-predict=autopeptideml.main:_predict',
            'autopeptideml=autopeptideml.main:_build_model',
            'autopeptideml-predict=autopeptideml.main:_predict',
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
    url='https://ibm.github.io/AutoPeptideML/',
    version='1.0.6',
    zip_safe=False,
)
