#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
readme = (this_directory / "README.md").read_text()

# with open('README.md', 'r') as readme_file:
#     readme = readme_file.read()

requirements = [ 
    'alembic>=1.11.1',
    'beautifulsoup4>=4.12.2',
    'certifi>=2023.7.22',
    'charset-normalizer>=3.2.0',
    'cmaes>=0.10.0',
    'colorlog>=6.7.0',
    'contourpy>=1.1.0',
    'cycler>=0.11.0',
    'dill>=0.3.6',
    'filelock>=3.12.2',
    'fonttools>=4.41.0',
    'fsspec>=2023.6.0',
    'gdown>=4.7.1',
    'graph-part>=1.0.2',
    'huggingface-hub>=0.16.4',
    'idna>=3.4',
    'Jinja2>=3.1.2',
    'joblib>=1.3.1',
    'kiwisolver>=1.4.4',
    'Mako>=1.2.4',
    'MarkupSafe>=2.1.3',
    'matplotlib>=3.7.2',
    'mpmath>=1.3.0',
    'networkx>=3.1',
    'numpy>=1.25.1',
    'optuna>=3.2.0',
    'packaging>=23.1',
    'pandarallel>=1.6.5',
    'pandas>=2.0.3',
    'Pillow',
    'psutil>=5.9.5',
    'pyparsing>=3.0.9',
    'PySocks>=1.7.1',
    'python-dateutil>=2.8.2',
    'pytz>=2023.3',
    'PyYAML>=6.0.1',
    'regex>=2023.6.3',
    'requests>=2.31.0',
    'safetensors>=0.3.1',
    'scikit-learn>=1.3.0',
    'scikit-plot>=0.3.7',
    'scipy>=1.11.1',
    'six>=1.16.0',
    'soupsieve>=2.4.1',
    'SQLAlchemy>=2.0.19',
    'sympy>=1.12',
    'threadpoolctl>=3.2.0',
    'tokenizers>=0.13.3',
    'torch>=2.0.1',
    'tqdm>=4.65.0',
    'transformers>=4.31.0',
    'typing_extensions>=4.7.1',
    'tzdata>=2023.3',
    'urllib3>=2.0.3',
    'lightgbm>=4.0.0',
    'mdpdf'
]

test_requirements = requirements

setup(
    author="Raul Fernandez-Diaz",
    author_email='raulfd@ibm.com',
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    description="Computational Pipeline for the Automatised Development of Peptide Bioactivity Prediction Models",
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
    version='0.1.0',
    zip_safe=False,
)
