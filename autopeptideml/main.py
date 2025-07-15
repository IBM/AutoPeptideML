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
    apml.preprocess_data(config['pipeline'], n_jobs=cpu_count())

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

    if not osp.isfile(metadata_path):
        raise RuntimeError("No metadata file was found. Check you're using the correct AutoPeptideML version.",
                           "Try: pip install autopeptideml<=2.0.1",
                           "Alternatively, check that the result_dir path is properly formatted.")
    if not osp.isdir(ensemble_path):
        raise RuntimeError("No ensemble path was found in result_dir.  Check you're using the correct AutoPeptideML version.",
                           "Try: pip install autopeptideml<=2.0.1",
                           "Alternatively, check that the result_dir path is properly formatted.")

    metadata = yaml.safe_load(open(metadata_path))
    print(f"Using model created on {metadata['start-time']} with AutoPeptideML v.{metadata['autopeptideml-version']}")
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
        preds = preds[:, 1]
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



# @app.command()
# def build_model(config_path: Optional[str] = None):
#     """
#     Build a machine learning model based on the provided configuration. If no configuration is provided
#     the configuration helper will prompt you for more details about the job you want to run.

#     Args:
#         config_path (str, optional): Path to the configuration file. Defaults to None.

#     Returns:
#         None
#     """
#     if config_path is not None:
#         config = yaml.safe_load(open(config_path))
#         mssg = f"| AutoPeptideML v.{__version__} |"
#         print("-"*(len(mssg)))
#         print(mssg)
#         print("-"*(len(mssg)))

#     else:
#         config_path = prepare_config()
#         config = yaml.safe_load(open(config_path))
#     print("** Model Builder **")
#     apml = AutoPeptideML(config)
#     db = apml.get_database()
#     reps = apml.get_reps()
#     test = apml.get_test()
#     models = apml.run_hpo()
#     r_df = apml.run_evaluation(models)
#     apml.save_experiment(save_reps=True, save_test=False)
#     print(r_df)


# @app.command()
# def prepare_config() -> dict:
#     mssg = f"| AutoPeptideML v.{__version__} |"
#     print("-"*(len(mssg)))
#     print(mssg)
#     print("-"*(len(mssg)))
#     print("** Experiment Builder **")
#     print("Please, answer the following questions to design your experiment.")

#     config_path = config_helper()
#     return config_path


# @app.command()
# def predict(experiment_dir: str, features_path: str, feature_field: str,
#             output_path: str = 'apml_predictions.csv'):
#     config_path = osp.join(experiment_dir, 'config.yml')
#     if not osp.exists(config_path):
#         raise FileNotFoundError("Configuration file was not found in experiment dir.")
#     config = yaml.safe_load(open(config_path))
#     apml = AutoPeptideML(config)
#     df = pd.read_csv(features_path)
#     results_df = apml.predict(
#         df, feature_field=feature_field,
#         experiment_dir=experiment_dir, backend='onnx'
#     )
#     results_df.to_csv(output_path, index=False, float_format="%.3g")


def _main():
    app()


if __name__ == "__main__":
    _main()
