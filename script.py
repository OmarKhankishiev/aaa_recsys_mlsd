import argparse
import warnings

import mlflow

from model.config import EXPERIMENT_NAME, MLFLOW_ADDRESS, NOTEBOOK_PARAMS
from model.run import (
    baseline_LFM,
    baseline_top_popular,
    best_model,
    optuna_ALS_search,
    optuna_LFM_search,
    run_ALS,
    run_LFM,
)
from model.utils import prepare_data, seed_everything

warnings.filterwarnings("ignore", category=RuntimeWarning)


MODELS = {
    # Baselines
    "baseline_LFM": baseline_LFM,
    "baseline_top_popular": baseline_top_popular,
    # Custom Models
    "LFM": run_LFM,
    "ALS": run_ALS,
    # Search best params
    "optuna_LFM_search": optuna_LFM_search,
    "optuna_ALS_search": optuna_ALS_search,
    # Best params from previous task
    "best_model": best_model,
}


def main(experiment_name, run_name, model_name, optimizer_name):
    seed_everything(NOTEBOOK_PARAMS["seed"])
    print(f"You choosed Model named {model_name}.")
    run_model = MODELS[model_name]
    data = prepare_data()

    mlflow.set_tracking_uri(MLFLOW_ADDRESS)
    mlflow.set_experiment(experiment_name)

    if model_name == "LFM":
        run_model(data, optimizer_name=optimizer_name)

    run_model(data)

    print("Successfully completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RecSys models.")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=EXPERIMENT_NAME,
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--run_name", type=str, default="best_model", help="Name of the run."
    )
    parser.add_argument(
        "--model_name", type=str, default="best_model", help="Model to run."
    )
    parser.add_argument(
        "--optimizer_name", type=str, default="Adam", help="Name of the optimizer."
    )

    args = parser.parse_args()

    main(args.experiment_name, args.run_name, args.model_name, args.optimizer_name)
