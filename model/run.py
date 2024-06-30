import json
import os
import random

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.config import NOTEBOOK_PARAMS
from model.utils import (
    ALSModel,
    LatentFactorModel,
    RecDataset,
    calculate_hitrate,
    calculate_precision,
    collate_fn,
    create_loader,
    get_preds,
    plot,
    seed_everything,
)


def run_LFM(
    data: tuple,
    batch_size: int = 30_000,
    n_negatives: int = 4,
    edim: int = 32,
    n_epochs: int = 3,
    optimizer_name: str = "Adam",
    learning_rate: float = 1,
    n_trials: int = 0,
    run_name: str = "lfm",
) -> float:
    (
        df_train,
        df_val,
        node2name,
        user2seen,
        user_indes,
        index2user_id,
        node_indes,
        index2node,
    ) = data

    run_params = {
        "batch_size": batch_size,
        "n_negatives": n_negatives,
        "edim": edim,
        "n_epochs": n_epochs,
        "optimizer_name": optimizer_name,
        "learning_rate": learning_rate,
        "n_workers": 0,
        "K": 30,
        "n_trials": n_trials,
    }

    dataloader = create_loader(df_train, df_val, user2seen, run_params)
    model = LatentFactorModel(
        run_params["edim"],
        user_indes,
        node_indes,
    )
    optimizers = {
        "Adam": torch.optim.Adam(model.parameters(), run_params["learning_rate"]),
        "SGD": torch.optim.SGD(model.parameters(), run_params["learning_rate"]),
        "RMSprop": torch.optim.RMSprop(model.parameters(), run_params["learning_rate"]),
        "AdamW": torch.optim.AdamW(model.parameters(), run_params["learning_rate"]),
        "Adagrad": torch.optim.Adagrad(model.parameters(), run_params["learning_rate"]),
    }
    optimizer = optimizers[run_params["optimizer_name"]]

    print(run_params)

    epoch_bar = tqdm(range(run_params["n_epochs"]))
    metrics = {"losses": [], "hitrate": [], "precision": []}

    for epoch in epoch_bar:
        metrics["loader_losses"] = []
        model.train()
        for (users, items, labels) in dataloader:
            optimizer.zero_grad()
            logits = model(users, items)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels,
            )
            loss.backward()
            optimizer.step()

            loader_loss = loss.item()
            metrics["loader_losses"].append(loader_loss)

        # Metrics
        with torch.no_grad():
            model.eval()
            df_preds = get_preds(model, df_val, run_params)
            epoch_loss = sum(metrics["loader_losses"]) / len(metrics["loader_losses"])
            hitrate = calculate_hitrate(df_preds, run_params["K"])
            precision = calculate_precision(df_preds, run_params["K"])

        del df_preds

        metrics["losses"].append(epoch_loss)
        metrics["hitrate"].append(hitrate)
        metrics["precision"].append(precision)

        epoch_bar.set_description(f"EPOCH LOSS = {epoch_loss:.4f}")
        fig = plot(metrics)

    with mlflow.start_run(run_name=run_name):
        for param, value in NOTEBOOK_PARAMS.items():
            mlflow.log_param(param, value)

        for param, value in run_params.items():
            mlflow.log_param(param, value)

        with torch.no_grad():
            model.eval()
            df_preds = get_preds(model, df_val, run_params)

        hitrate = calculate_hitrate(df_preds, run_params["K"])
        precision = calculate_precision(df_preds, run_params["K"])
        print(f"Hitrate: {hitrate:.4f} | Precision: {precision:.4f}")

        mlflow.log_metrics(
            {
                "hitrate": hitrate,
                "precision": precision,
            }
        )
        mlflow.log_artifact(NOTEBOOK_PARAMS["notebook_path"])
        mlflow.log_figure(fig, "metrics.png")

    return precision


def run_top_popular(data: tuple, run_params: dict, run_name: str) -> float:
    (
        df_train,
        df_val,
        node2name,
        user2seen,
        user_indes,
        index2user_id,
        node_indes,
        index2node,
    ) = data

    top_popular = (
        df_train[["node_index"]]
        .assign(v=1)
        .groupby("node_index")
        .count()
        .reset_index()
        .sort_values(by="v")
        .tail(run_params["K"])["node_index"]
        .values
    )

    users = df_val["user_index"].unique()
    df_preds_top_poplular = pd.DataFrame(
        {
            "node_index": [list(top_popular) for i in users],
            "user_index": users,
            "rank": [[j for j in range(0, run_params["K"])] for i in range(len(users))],
        }
    )

    df_preds_top_poplular = df_preds_top_poplular.explode(
        [
            "node_index",
            "rank",
        ]
    ).merge(
        df_val[["user_index", "node_index"]].assign(relevant=1).drop_duplicates(),
        on=["user_index", "node_index"],
        how="left",
    )
    df_preds_top_poplular["relevant"] = df_preds_top_poplular["relevant"].fillna(0)

    with mlflow.start_run(run_name=run_name):
        for param, value in NOTEBOOK_PARAMS.items():
            mlflow.log_param(param, value)

        for param, value in run_params.items():
            mlflow.log_param(param, None)

        hitrate = calculate_hitrate(df_preds_top_poplular, run_params["K"])
        precision = calculate_precision(df_preds_top_poplular, run_params["K"])
        print(f"Hitrate: {hitrate:.4f} | Precision: {precision:.4f}")

        mlflow.log_metrics(
            {
                "hitrate": hitrate,
                "precision": precision,
            }
        )
        mlflow.log_artifact(NOTEBOOK_PARAMS["notebook_path"])


def baseline_LFM(data, run_name: str = "baseline_LFM_pipeline") -> float:
    run_params = {
        "batch_size": 50_000,
        "n_negatives": 5,
        "edim": 128,
        "n_epochs": 10,
        "optimizer_name": "Adam",
        "learning_rate": 1,
        "n_workers": 0,
        "n_trials": 0,
        "K": 30,
    }
    seed_everything(NOTEBOOK_PARAMS["seed"])

    precision = run_LFM(
        data=data,
        batch_size=run_params["batch_size"],
        n_negatives=run_params["n_negatives"],
        edim=run_params["edim"],
        n_epochs=run_params["n_epochs"],
        optimizer_name=run_params["optimizer_name"],
        learning_rate=run_params["learning_rate"],
        n_trials=run_params["n_trials"],
        run_name=run_name,
    )
    return precision


def baseline_top_popular(
    data,
    run_name: str = "baseline_top_popular_pipeline",
) -> float:
    run_params = {
        "K": 30,
    }
    seed_everything(NOTEBOOK_PARAMS["seed"])
    precision = run_top_popular(
        data,
        run_params,
        run_name=run_name,
    )
    return precision


def best_model(data, run_name: str = "best_model_pipeline") -> float:
    run_params = {
        "factors": 177,
        "regularization": 2.699644745630464,
        "iterations": 27,
        "alpha": 3.7118064519704,
        "n_negatives": 6,
        "n_workers": 0,
        "K": 30,
        "n_trials": 0,
    }
    seed_everything(NOTEBOOK_PARAMS["seed"])
    precision = run_ALS(
        data=data,
        factors=run_params["factors"],
        regularization=run_params["regularization"],
        iterations=run_params["iterations"],
        n_negatives=run_params["n_negatives"],
        alpha=run_params["alpha"],
        n_trials=run_params["n_trials"],
        run_name=run_name,
    )
    return precision


def optuna_LFM_search(data, run_name: str = "optuna_LFM_pipeline") -> float:
    n_trials = 3

    def objective(trial) -> float:
        seed_everything(NOTEBOOK_PARAMS["seed"])
        batch_size = trial.suggest_categorical(
            "batch_size",
            [5000, 10000, 30000, 50000],
        )
        learning_rate = trial.suggest_float("learning_rate", 1e-2, 3)
        n_negatives = trial.suggest_int("n_negatives", 1, 3)
        edim = trial.suggest_int("edim", 32, 64)
        n_epochs = trial.suggest_int("n_epochs", 2, 5)
        optimizer_name = trial.suggest_categorical(
            "optimizer_name",
            [
                "Adam",
                "SGD",
                "RMSprop",
                "AdamW",
                "Adagrad",
            ],
        )

        precision = run_LFM(
            data=data,
            batch_size=batch_size,
            n_negatives=n_negatives,
            edim=edim,
            n_epochs=n_epochs,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            n_trials=n_trials,
            run_name=run_name,
        )
        return precision

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)


def run_ALS(
    data,
    factors,
    regularization,
    iterations,
    n_trials,
    n_negatives,
    alpha,
    run_name,
) -> float:
    (
        df_train,
        df_val,
        node2name,
        user2seen,
        user_indes,
        index2user_id,
        node_indes,
        index2node,
    ) = data

    run_params = {
        "factors": factors,
        "regularization": regularization,
        "iterations": iterations,
        "alpha": alpha,
        "n_negatives": n_negatives,
        "n_workers": 0,
        "K": 30,
        "n_trials": n_trials,
    }

    print(run_params)
    model = ALSModel(
        factors=run_params["factors"],
        regularization=run_params["regularization"],
        iterations=run_params["iterations"],
        alpha=run_params["alpha"],
    )

    train_dataset = RecDataset(
        df_train["user_index"].values,
        df_train["node_index"].values,
        user2seen,
    )
    dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=run_params["n_workers"],
        batch_size=len(train_dataset),
        collate_fn=lambda batch: collate_fn(
            batch=batch,
            num_negatives=run_params["n_negatives"],
            num_items=max(
                df_train["node_index"].max(),
                df_val["node_index"].max(),
            ),
        ),
    )
    users, items, labels = next(iter(dataloader))
    model.fit(users, items, labels, run_params["n_negatives"])

    with mlflow.start_run(run_name=run_name):
        for param, value in NOTEBOOK_PARAMS.items():
            mlflow.log_param(param, value)

        for param, value in run_params.items():
            mlflow.log_param(param, value)

        with torch.no_grad():
            model.eval()
            df_preds = get_preds(model, df_val, run_params)

        hitrate = calculate_hitrate(df_preds, run_params["K"])
        precision = calculate_precision(df_preds, run_params["K"])
        print(f"Hitrate: {hitrate:.4f} | Precision: {precision:.4f}")

        mlflow.log_metrics(
            {
                "hitrate": hitrate,
                "precision": precision,
            }
        )
        mlflow.log_artifact(NOTEBOOK_PARAMS["notebook_path"])

    return precision


def optuna_ALS_search(data, run_name: str = "optuna_ALS_pipeline") -> float:
    n_trials = 3

    def objective(trial) -> float:
        seed_everything(NOTEBOOK_PARAMS["seed"])
        factors = trial.suggest_int("factors", 10, 15)
        regularization = trial.suggest_float("regularization", 1e-3, 2)
        iterations = trial.suggest_int("iterations", 3, 7)
        n_negatives = trial.suggest_int("n_negatives", 1, 5)
        alpha = trial.suggest_float("alpha", 0.1, 10)

        precision = run_ALS(
            data=data,
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            n_negatives=n_negatives,
            alpha=alpha,
            n_trials=n_trials,
            run_name=run_name,
        )
        return precision

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
