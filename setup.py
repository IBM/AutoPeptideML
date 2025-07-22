#!/usr/bin/env python

"""The setup script."""
from setuptools import setup, find_packages
from pathlib import Path


this_file = Path(__file__).resolve()
this_directory = this_file.parent
readme = (this_directory / "README.md").read_text()

requirements = [
    'optuna',
    'scikit-learn',
    'typer',
    'tokenizers',
    'torch',
    'transformers',
    'lightgbm',
    'xgboost',
    'seaborn',
    # 'catboost',
    'hestia-good',
    'onnxmltools',
    'skl2onnx',
    'onnxruntime',
    'rdkit',
    'quarto',
    'mapchiral',
    'tabulate'
]


def get_files_in_dir(path: Path, base: Path) -> list:
    paths = []
    if path.is_file():
        if path.suffix == '.csv':
            return [None]
        return [path.relative_to(base)]
    elif path.is_dir():
        for subpath in path.iterdir():
            paths += get_files_in_dir(subpath, base)
    paths = set([p for p in paths if p is not None])
    return list(paths)


test_requirements = requirements
data_dir = this_directory / 'autopeptideml' / 'data'
files = get_files_in_dir(data_dir, this_directory)
files = [str(f) for f in files]

setup(
    author="Raul Fernandez-Diaz",
    author_email='raulfd@ibm.com',
    python_requires='>=3.9',
    classifiers=[
    ],
    description="AutoML system for building trustworthy peptide bioactivity predictors",
    entry_points={
        'console_scripts': [
            'apml=autopeptideml.main:_main',
            'autopeptideml=autopeptideml.main:_main'
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
    version='2.0.3',
    zip_safe=False,
)
