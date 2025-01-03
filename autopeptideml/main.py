import json
from multiprocessing import cpu_count
import os
import os.path as osp
import yaml

from typing import *

import pandas as pd
import typer

from .__init__ import __version__
from .autopeptideml import AutoPeptideML
from .utils.embeddings import RepresentationEngine

import typer
from multiprocessing import cpu_count


def build_model(
    dataset: str,
    outputdir: str = typer.Option("apml_result", help="Directory to save output results"),
    verbose: Annotated[bool, typer.Option("--verbose", help="Enable verbose output")] = False,
    threads: int = typer.Option(cpu_count(), help="Number of threads to use"),
    seed: int = typer.Option(1, help="Random seed for reproducibility"),
    plm: str = typer.Option("esm2-8m", help="PLM for computing peptide representations. Check GitHub Repository for available options."),
    plm_batch_size: int = typer.Option(64, help="Batch size for PLM"),
    plm_device: str = typer.Option(None, help="Device for PLM computation"),
    config: str = typer.Option("default_config", help="Configuration file"),
    autosearch: str = typer.Option("auto", help="Whether to search for negative peptides."),
    autosearch_tags: str = typer.Option("", help="Comma-separated list of positive tags to exclude from autosearch."),
    autosearch_proportion: float = typer.Option(1.0, help="Negative:positive proportion."),
    autosearch_db: str = typer.Option(None, help="Alternative database to draw negatives from."),
    model_save_backend: str = typer.Option("onnx", help="Backend for storing models. Options: `onnx` or `joblib`"),
    balance: Annotated[bool, typer.Option("--balance", help="Whether to oversample the underrepresented class.")] = True,
    test_partition: Annotated[bool, typer.Option("--test-partition", help="Whether to divide dataset into train/test splits.")] = True,
    test_threshold: float = typer.Option(0.3, help="Threshold for test partition."),
    test_size: float = typer.Option(0.2, help="Size of the test partition."),
    test_alignment: str = typer.Option("peptides", help="Alignment method for test partition."),
    splits: str = typer.Option(None, help="Splits configuration for dataset."),
    val_partition: Annotated[bool, typer.Option("--val_partition", help="Whether to divide dataset into train/validation folds.")] = True,
    val_method: str = typer.Option("random", help="Validation method."),
    val_alignment: str = typer.Option("peptides", help="Alignment method for validation."),
    val_threshold: float = typer.Option(0.5, help="Threshold for validation partition."),
    val_n_folds: int = typer.Option(10, help="Number of validation folds."),
    folds: str = typer.Option(None, help="Folds configuration for validation.")
) -> pd.DataFrame:
    """Builds and trains a predictive model using peptide datasets and a Protein Language Model (PLM).

    :param dataset: Path to the dataset to be used for training. If 'None', the dataset will not be curated.
    :type dataset: str
    :param outputdir: Directory where the results, configurations, and model outputs will be saved.
    :type outputdir: str, optional
    :param verbose: Enables verbose output if set to True. Logs progress and detailed outputs during execution.
    :type verbose: bool, optional
    :param threads: Number of threads to use for parallel processing.
    :type threads: int, optional
    :param seed: Random seed for reproducibility.
    :type seed: int, optional
    :param plm: Peptide Language Model (PLM) to use for computing peptide representations. Refer to the GitHub repository for supported PLM options.
    :type plm: str, optional
    :param plm_batch_size: Batch size for processing data through the PLM. Adjust based on available memory and dataset size.
    :type plm_batch_size: int, optional
    :param plm_device: Device (e.g., "cuda", "cpu", "cuda:0") for PLM computations. If None, the default device is used.
    :type plm_device: str, optional
    :param config: Path to a JSON configuration file or the name of a predefined configuration. Used for hyperparameter optimization and model training.
    :type config: str, optional
    :param autosearch: Determines whether to search for negative peptides. Use "auto" to automatically search when insufficient negatives exist or "True" to force search.
    :type autosearch: str, optional
    :param autosearch_tags: Comma-separated list of positive tags to exclude from the autosearch process.
    :type autosearch_tags: str, optional
    :param autosearch_proportion: Ratio of negative to positive samples to be maintained in the dataset.
    :type autosearch_proportion: float, optional
    :param autosearch_db: Alternative database to draw negative samples from.
    :type autosearch_db: str, optional
    :param model_save_backend: Backend for storing models. Options: 'onnx' or 'joblib'.
    :type model_save_backend: str, optional
    :param balance: Whether to oversample the underrepresented class.
    :type balance: bool, optional
    :param test_partition: Whether to divide the dataset into training and testing splits.
    :type test_partition: bool, optional
    :param test_threshold: Threshold value used for test partitioning.
    :type test_threshold: float, optional
    :param test_size: Proportion of the dataset to allocate for testing.
    :type test_size: float, optional
    :param test_alignment: Method for aligning the test partition.
    :type test_alignment: str, optional
    :param splits: Path to an existing directory containing predefined training and testing splits.
    :type splits: str, optional
    :param val_partition: Whether to partition the training data into validation folds.
    :type val_partition: bool, optional
    :param val_method: Method for generating validation folds.
    :type val_method: str, optional
    :param val_alignment: Method for aligning validation partitions.
    :type val_alignment: str, optional
    :param val_threshold: Threshold value used for validation partitioning.
    :type val_threshold: float, optional
    :param val_n_folds: Number of validation folds to generate.
    :type val_n_folds: int, optional
    :param folds: Path to an existing directory containing predefined validation folds.
    :type folds: str, optional
    :return: A DataFrame containing evaluation results of the trained model, including metrics such as accuracy, precision, recall, and F1 score.
    :rtype: pd.DataFrame

    Notes:
    - Creates an `apml_config.yaml` file in the output directory to save configuration details.
    - Generates or balances negative peptide samples based on input parameters.
    - Partitions the dataset into training, validation, and testing sets unless predefined splits are provided.
    - Performs hyperparameter optimization (HPO) and evaluates the best model.

    Example:
    >>> results = build_model("path/to/dataset.csv", verbose=True, plm_device="cuda:0")
    """
    AutoPeptideML._welcome()
    typer.echo(f"Dataset: {dataset}")
    typer.echo(f"Output Directory: {outputdir}")

    os.makedirs(outputdir, exist_ok=True)
    apml_config_path = os.path.join(
        outputdir, 'apml_config.yaml'
    )
    variables = locals()
    variables.update({'version': __version__})
    del variables['apml_config_path']
    with open(apml_config_path, "w") as yaml_file:
        yaml.dump(variables, yaml_file, default_flow_style=False,
                  sort_keys=True)

    re = RepresentationEngine(plm, plm_batch_size)
    if plm_device is not None:
        re.move_to_device(plm_device)
    apml = AutoPeptideML(verbose, threads, seed)

    if dataset != 'None':
        df = apml.curate_dataset(dataset, outputdir)
        if 'id' not in df.columns:
            df['id'] = df.index

    if ((autosearch == 'auto' and len(df[df.Y == 0]) < 1) or
       autosearch == 'True'):
        df = apml.autosearch_negatives(
            df,
            autosearch_tags.split(','),
            autosearch_proportion
        )
    if balance == 'True':
        df = apml.balance_saples(df)

    if test_partition and splits is None:
        datasets = apml.train_test_partition(
            df=df,
            threshold=test_threshold,
            test_size=test_size,
            denominator='longest',
            alignment=test_alignment,
            outputdir=osp.join(outputdir, 'splits')
        )
    else:
        datasets = {
            'train': pd.read_csv(os.path.join(splits, 'train.csv')),
            'test': pd.read_csv(os.path.join(splits, 'test.csv'))
        }

    if val_partition and folds is None:
        folds = apml.train_val_partition(
            df=datasets['train'],
            method=val_method,
            threshold=val_threshold,
            alignment=val_alignment,
            n_folds=val_n_folds,
            outputdir=os.path.join(outputdir, 'folds')
        )
    else:
        folds = [
            {'train': pd.read_csv(os.path.join(folds, f'train_{i}.csv')),
             'val': pd.read_csv(os.path.join(folds, f'val_{i}.csv'))}
            for i in range(val_n_folds)
        ]

    id2rep = apml.compute_representations(datasets, re)
    if not config.endswith('.json'):
        config = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'data', 'configs', config + '.json'
        )

    model = apml.hpo_train(
        json.load(open(config)),
        datasets['train'],
        id2rep,
        folds,
        outputdir,
    )
    results = apml.evaluate_model(
        model,
        datasets['test'],
        id2rep,
        outputdir
    )
    apml.save_models(
        best_model=model,
        outputdir=osp.join(outputdir, 'ensemble'),
        id2rep=id2rep,
        backend=model_save_backend
    )
    if verbose is True:
        print(results)
    return results


def predict(
    dataset: str,
    ensemble: str = typer.Option(None, help="Path to directory with previous APML results."),
    outputdir: str = typer.Option("apml_predictions", help="Directory to save prediction results."),
    verbose: str = typer.Option("True", help="Enable verbose output."),
    threads: int = typer.Option(cpu_count(), help="Number of threads to use."),
    plm: str = typer.Option("esm2-8m", help="PLM for computing peptide representations. Check GitHub Repository for available options."),
    plm_batch_size: int = typer.Option(12, help="Batch size for PLM."),
    plm_device: str = typer.Option(None, help="Device where the representations should be computed."),
    model_save_backend: str = typer.Option("onnx", help="Backend for storing models. Options: `onnx` or `joblib`"),
) -> pd.DataFrame:
    """
    Predicts peptide representations and outputs predictions using a pre-trained Peptide Language Model (PLM).

    This function takes a dataset and processes it through a specified PLM to compute peptide representations. It integrates with the AutoPeptideML (APML) framework to curate the dataset, perform predictions, and optionally use an ensemble of previous results.

    Parameters:
    ----------
    dataset : str
        Path to the dataset to be processed. The dataset should be in a format compatible with APML.

    ensemble : str, optional
        Path to a directory containing previous APML results for ensemble predictions. If `None`, no ensemble is used.
        Default is `None`.

    outputdir : str, optional
        Directory where the prediction results will be saved. If the directory does not exist, it will be created.
        Default is `"apml_predictions"`.

    verbose : str, optional
        Controls the verbosity of the output. Set to `"True"` for detailed logs or `"False"` for minimal logs.
        Default is `"True"`.

    threads : int, optional
        Number of threads to use for parallel processing. Defaults to the number of CPU cores available.

    plm : str, optional
        The Peptide Language Model (PLM) to use for computing peptide representations. Refer to the GitHub repository for supported PLM options.
        Default is `"esm2-8m"`.

    plm_batch_size : int, optional
        Batch size for processing data through the PLM. Adjust based on available memory and dataset size.
        Default is `12`.

    device : str, optional
        Specifies the device (e.g., `"cuda"`, `"cpu"`, `"cuda:0"`) for PLM computations. If `None`, the default device is used.
        Default is `None`.

    Returns:
    -------
    pd.DataFrame
        A pandas DataFrame containing the predictions. The structure of the DataFrame depends on the dataset and the APML framework's prediction output.

    Notes:
    -----
    - The function initializes a `RepresentationEngine` for computing peptide representations and moves it to the specified device if provided.
    - The dataset is curated using the APML framework, and predictions are generated based on the provided PLM and ensemble (if applicable).
    - Ensure the PLM and APML dependencies are correctly installed and configured.

    Examples:
    --------
    ```python
    # Predict using the default PLM and save results to the default output directory
    df = predict("path/to/dataset.csv")

    # Predict using a specific ensemble directory and a custom device
    df = predict(
        dataset="path/to/dataset.csv",
        ensemble="path/to/ensemble",
        device="cuda:0"
    )
    """
    re = RepresentationEngine(plm, plm_batch_size)
    if plm_device is not None:
        re.move_to_device(plm_device)
    apml = AutoPeptideML(verbose, threads, 1)
    df = apml.curate_dataset(dataset, outputdir)
    return apml.predict(df, re, ensemble, outputdir,
                        backend=model_save_backend)


def _build_model():
    typer.run(build_model)


def _predict():
    typer.run(predict)
