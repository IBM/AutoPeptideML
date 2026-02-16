import os.path as osp
import yaml

from multiprocessing import cpu_count
from typing import Optional

import typer

from .apml import AutoPeptideML, __version__
from .utils.config import config_helper
from .utils.dataset_parsing import read_data

app = typer.Typer()


def welcome():
    mssg = f"AutoPeptideML v.{__version__}\n"
    mssg += "By Raul Fernandez-Diaz"
    max_width = max([len(line) for line in mssg.split('\n')])

    print("-" * (max_width + 4))
    for line in mssg.split('\n'):
        print("| " + line + " " * (max_width - len(line)) + " |")
    print("-" * (max_width + 4))


@app.command()
def build_model(
    outdir: Optional[str] = 'apml-result',
    config_path: Optional[str] = None
) -> AutoPeptideML:
    """
    Build a machine learning model based on a specified configuration.

    This command initializes an AutoPeptideML model using configuration details
    loaded either from a provided YAML config file or from a default path
    within the output directory. It preprocesses the data, optionally samples
    negatives, trains the model ensemble, and generates a report.

    :param outdir: Output directory where results and intermediate files will be saved.
                   Default is 'apml-result'.
                   This directory will also be used to look for the default configuration
                   file if `config_path` is not provided.
                   Type: str

    :param config_path: Path to the YAML configuration file.
                        If None, it defaults to `<outdir>/setup-config.yml`.
                        The config file should define keys such as:
                        - `datasets`: Dataset details, including paths and label/feature fields.
                        - `pipeline`: Data preprocessing steps.
                        - `n-jobs`: Number of parallel jobs (-1 for all CPUs).
                        - `task`: Task type, e.g., 'classification', 'regression'.
                        - `reps`: List of representation types to use.
                        - `models`: List of model architectures to build.
                        - `n-trials`: Number of hyperparameter optimization trials.
                        - `device`: Computing device, e.g., 'cpu' or 'cuda'.
                        Type: str or None

    :returns: An instance of the trained AutoPeptideML model.
    :rtype: AutoPeptideML

    :raises FileNotFoundError: If the config file or dataset files specified in the config are missing.
    :raises yaml.YAMLError: If the config file is not a valid YAML.

    **Example values:**

    - `outdir`: `'results/my_experiment'`
    - `config_path`: `'configs/exp1-config.yml'` or `None` to use default
    - `config['task']`: `'classification'`, `'regression'`
    - `config['n-jobs']`: `-1` (use all CPUs), or a positive integer like `4`
    - `config['device']`: `'cpu'`, `'cuda'`
    """
    welcome()
    print("\nModel builder\n")
    if config_path is None:
        config_path = osp.join(outdir, 'setup-config.yml')
        config = config_helper(config_path)
    else:
        config = yaml.safe_load(open(config_path))

    df = read_data(config['datasets']['main']['path'])
    if 'Assume' in config['datasets']['main']['label-field']:
        apml = AutoPeptideML(
            data=df[config['datasets']['main']['feat-fields']].tolist(),
            outputdir=outdir,
        )
    else:
        apml = AutoPeptideML(
            data=df,
            outputdir=outdir,
            sequence_field=config['datasets']['main']['feat-fields'],
            label_field=config['datasets']['main']['label-field']
        )

    if 'neg-db' in config['datasets']:
        apml.sample_negatives(
            target_db=config['datasets']['neg-db']['path'],
            activities_to_exclude=config['datasets']['neg-db']['activities-to-exclude'],
        )
    if config['n-jobs'] == -1:
        config['n-jobs'] = cpu_count()

    apml.build_models(
        task=config['task'],
        ensemble=False,
        reps=config['reps'],
        models=config['models'],
        split_strategy='min',
        n_trials=config['n-trials'],
        device=config['device'],
        random_state=1,
        n_jobs=config['n-jobs']
    )
    apml.create_report()
    return apml


@app.command()
def prepare_config(config_path: str) -> dict:
    """
    Prepare and load the configuration for an AutoPeptideML experiment.

    This command loads a YAML configuration file that defines the parameters
    for building or running a model. If the provided `config_path` does not
    end with `.yml`, the suffix is automatically appended.

    :param config_path: Path to the YAML configuration file.
                        The file should contain experiment setup details such as:
                        - Dataset paths and fields
                        - Pipeline preprocessing steps
                        - Model hyperparameters and training options
                        - Device settings (e.g., 'cpu', 'cuda', 'mps')
                        Must be a string ending with `.yml` (appended automatically if missing).
                        Example: `'configs/my_experiment.yml'`

    :returns: The parsed configuration dictionary loaded from the YAML file.
    :rtype: dict

    :raises FileNotFoundError: If the configuration file does not exist at the given path.
    :raises yaml.YAMLError: If the configuration file is not a valid YAML file.

    **Example values:**

    - `config_path`: `'config.yml'`, `'experiment/setup-config'` (appends `.yml` automatically)
    """
    welcome()
    print("\nPrepare configuration\n")
    if not config_path.endswith('.yml'):
        config_path += '.yml'
    config = config_helper(config_path=config_path)
    return config


@app.command()
def predict(result_dir: str, features_path: str,
            feature_field: Optional[str] = None,
            output_path: Optional[str] = 'predictions.tsv',
            n_jobs: Optional[int] = -1,
            device: Optional[str] = 'cpu'):
    """
    Perform prediction using a trained AutoPeptideML ensemble model.

    This command loads a trained ensemble model from the specified `result_dir`,
    reads input features from `features_path`, computes molecular or sequence
    representations as needed, and generates prediction scores along with uncertainty.

    :param result_dir: Directory containing the trained model results.
                       Must include `metadata.yml` and `ensemble/` subdirectory.
                       Example: `'apml-result'`
                       Type: str

    :param features_path: Path to the input dataset file containing features to predict on.
                          Supported formats depend on `read_data` function (commonly CSV/TSV).
                          Example: `'data/new_peptides.csv'`
                          Type: str

    :param feature_field: Name of the column in the input data to use for feature extraction.
                          If None, the function attempts to infer it from columns
                          such as `'sequence'`, `'smiles'`, or `'SMILES'`.
                          Type: str or None
                          Possible values:
                          - `'sequence'`
                          - `'smiles'`
                          - `'SMILES'`
                          - Any other column name containing molecular or sequence data

    :param output_path: Path where the prediction results will be saved.
                        Defaults to `'predictions.tsv'`.
                        Output file is tab-separated and contains original data plus
                        columns for `'preds'` (prediction scores) and `'uncertainty'`.
                        Type: str

    :param n_jobs: Number of parallel jobs to use for processing.
                   Defaults to `-1` to use all available CPUs.
                   Type: int
                   Possible values:
                   - `-1` (use all CPUs)
                   - Positive integer specifying number of jobs (e.g., 4)

    :param device: Device to run the prediction computations on.
                   Defaults to `'cpu'`.
                   Type: str
                   Possible values:
                   - `'cpu'`
                   - `'cuda'` (GPU acceleration, if available)
                   - `'mps'` (MPS acceleration, if available)

    :returns: A pandas DataFrame containing the input features with added columns:
              - `preds`: predicted probabilities or values.
              - `uncertainty`: uncertainty estimate of predictions.
    :rtype: pandas.DataFrame

    :raises RuntimeError: If required model metadata or ensemble directory is missing.
    :raises FileNotFoundError: If `features_path` does not exist or is unreadable.

    **Example usage:**

    ```python
    predict(
        result_dir='apml-result',
        features_path='data/peptides.csv',
        feature_field='sequence',
        output_path='my_predictions.tsv',
        n_jobs=4,
        device='cuda'
    )
    ```
    """
    from .pipeline import get_pipeline
    from .reps import PLMs, CLMs
    from .train.architectures import VotingEnsemble
    from .utils.dataset_parsing import read_data

    welcome()
    print("\nPrediction\n")
    metadata_path = osp.join(result_dir, 'metadata.yml')
    ensemble_path = osp.join(result_dir, 'ensemble')

    if n_jobs == -1:
        n_jobs = cpu_count()

    if osp.isfile(metadata_path):
        metadata = yaml.safe_load(open(metadata_path))
        print(f"Using model created on {metadata['start-time']} with AutoPeptideML v.{metadata['autopeptideml-version']}")

        # raise RuntimeError("No metadata file was found. Check you're using the correct AutoPeptideML version.",
        #                    "Try: pip install autopeptideml<=2.0.1",
        #                    "Alternatively, check that the result_dir path is properly formatted.")
    if not osp.isdir(ensemble_path):
        raise RuntimeError("No ensemble path was found in result_dir.  Check you're using the correct AutoPeptideML version.",
                           "Try: pip install autopeptideml<=2.0.1",
                           "Alternatively, check that the result_dir path is properly formatted.")

    ensemble = VotingEnsemble.load(ensemble_path)
    df = read_data(features_path)
    if feature_field is None:
        if 'sequence' in df:
            feature_field = 'sequence'
        if 'smiles' in df:
            feature_field = 'smiles'
        if 'SMILES' in df:
            feature_field = 'SMILES'

    prot, mol = False, False
    for rep in ensemble.reps:
        if rep in PLMs:
            prot = True
        elif rep in CLMs:
            mol = True
        elif 'fp' in rep:
            mol = True

    if prot:
        pipe = get_pipeline('to-sequences')
        df['pipe-seq'] = pipe(df[feature_field],
                              n_jobs=n_jobs,
                              verbose=True)
    if mol:
        pipe = get_pipeline('to-smiles')
        df['pipe-mol'] = pipe(df[feature_field],
                              n_jobs=n_jobs,
                              verbose=True)
    x = {}
    for rep in ensemble.reps:
        if rep in x:
            continue
        if rep in PLMs:
            from .reps.lms import RepEngineLM
            repengine = RepEngineLM(rep)
            repengine.move_to_device(device)
            x[rep] = repengine.compute_reps(
                df['pipe-seq'], verbose=True,
                batch_size=12 if repengine.get_num_params() > 2e7 else 64)

        elif rep in CLMs:
            from .reps.lms import RepEngineLM
            repengine = RepEngineLM(rep)
            repengine.move_to_device(device)
            x[rep] = repengine.compute_reps(
                df['pipe-mol'], verbose=True,
                batch_size=12 if repengine.get_num_params() > 2e7 else 64)

        elif 'fp' in rep:
            from .reps.fps import RepEngineFP

            repengine = RepEngineFP(rep=rep.split('-')[0],
                                    nbits=int(rep.split('-')[2]),
                                    radius=int(rep.split('-')[1]))
            x[rep] = repengine.compute_reps(
                df['pipe-mol'], verbose=True, batch_size=100)

    try:
        preds, std = ensemble.predict_proba(x)
        # preds = preds
    except AttributeError:
        preds, std = ensemble.predict(x)
    if 'pipe-seq' in df:
        df.drop(columns=['pipe-seq'], inplace=True)
    if 'pipe-mol' in df:
        df.drop(columns=['pipe-mol'], inplace=True)

    df['preds'] = preds
    df['uncertainty'] = std

    df.to_csv(output_path, index=False)
    return df


def _main():
    app()


if __name__ == "__main__":
    _main()
