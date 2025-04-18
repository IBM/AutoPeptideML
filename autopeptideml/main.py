import os
import os.path as osp
import time
import yaml

from typing import *

import pandas as pd
import typer

from .autopeptideml import AutoPeptideML, __version__
from .config import config_helper


app = typer.Typer()


@app.command()
def build_model(config_path: Optional[str] = None):
    """
    Build a machine learning model based on the provided configuration. If no configuration is provided
    the configuration helper will prompt you for more details about the job you want to run.

    Args:
        config_path (str, optional): Path to the configuration file. Defaults to None.

    Returns:
        None
    """
    if config_path is not None:
        config = yaml.safe_load(open(config_path))
        mssg = f"| AutoPeptideML v.{__version__} |"
        print("-"*(len(mssg)))
        print(mssg)
        print("-"*(len(mssg)))

    else:
        config_path = prepare_config()
        config = yaml.safe_load(open(config_path))
    print("** Model Builder **")
    apml = AutoPeptideML(config)
    db = apml.get_database()
    reps = apml.get_reps()
    test = apml.get_test()
    models = apml.run_hpo()
    r_df = apml.run_evaluation(models)
    apml.save_experiment(save_reps=True, save_test=False)
    print(r_df)


@app.command()
def prepare_config() -> dict:
    mssg = f"| AutoPeptideML v.{__version__} |"
    print("-"*(len(mssg)))
    print(mssg)
    print("-"*(len(mssg)))
    print("** Experiment Builder **")
    print("Please, answer the following questions to design your experiment.")

    config_path = config_helper()
    return config_path


@app.command()
def predict(experiment_dir: str, features_path: str, feature_field: str,
            output_path: str = 'apml_predictions.csv'):
    config_path = osp.join(experiment_dir, 'config.yml')
    if not osp.exists(config_path):
        raise FileNotFoundError("Configuration file was not found in experiment dir.")
    config = yaml.safe_load(open(config_path))
    apml = AutoPeptideML(config)
    df = pd.read_csv(features_path)
    results_df = apml.predict(
        df, feature_field=feature_field,
        experiment_dir=experiment_dir, backend='onnx'
    )
    results_df.to_csv(output_path, index=False, float_format="%.3g")


def _main():
    app()


if __name__ == "__main__":
    _main()
