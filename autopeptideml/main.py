import os.path as osp
import yaml

from multiprocessing import cpu_count
from typing import Optional

import typer

from .apml import AutoPeptideML, __version__
from .config import config_helper
from .utils.dataset_parsing import read_data

app = typer.Typer()


def welcome():
    mssg = f"AutoPeptideML v.{__version__}\n"
    mssg += "By Raul Fernandez-Diaz"
    max_width = max([len(line) for line in mssg.split('\n')])

    print("-" * (max_width + 3))
    for line in mssg.split('\n'):
        print("| " + line + " " * (max_width - len(line)) + " |")
    print("-" * (max_width + 3))


@app.command()
def build_model(
    outdir: Optional[str] = 'apml-result',
    config_path: Optional[str] = None
):
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


@app.command()
def prepare_config(config_path: str):
    welcome()
    print("\nPrepare configuration\n")
    if not config_path.endswith('.yml'):
        config_path += '.yml'
    config = config_helper(config_path=config_path)




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
