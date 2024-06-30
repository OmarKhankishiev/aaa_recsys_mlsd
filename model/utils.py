import json
import os
import random

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.config import NOTEBOOK_PARAMS


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RecDataset(Dataset):
    def __init__(self, users, items, item_per_users):
        self.users = users
        self.items = items
        self.item_per_users = item_per_users

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx: int):
        user = self.users[idx]
        return (
            torch.tensor(user),
            torch.tensor(self.items[idx]),
            self.item_per_users[user],
        )


class LatentFactorModel(nn.Module):
    def __init__(self, edim, user_indexes, node_indexes):
        super(LatentFactorModel, self).__init__()
        self.edim = edim
        self.users = nn.Embedding(max(user_indexes) + 1, edim)
        self.items = nn.Embedding(max(node_indexes) + 1, edim)

    def forward(self, users, items):
        user_embedings = self.users(users).reshape(-1, self.edim)
        item_embedings = self.items(items)
        # (n_users; edim) + (batch; n_items; edim) = (batch, n_items)
        res = torch.einsum("be,bne->bn", user_embedings, item_embedings)
        return res

    def pred_top_k(self, users, K: int):
        user_embedings = self.users(users).reshape(-1, self.edim)
        item_embedings = self.items.weight
        # (n_users; edim) + (n_items; edim) = (n_users, n_items)
        res = torch.einsum("ue,ie->ui", user_embedings, item_embedings)
        return torch.topk(res, K, dim=1)


class ALSModel(nn.Module):
    def __init__(self, factors, regularization, iterations, alpha):
        super(ALSModel, self).__init__()
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            alpha=alpha,
        )
        self.matrix = None
        self.is_train = False

    def fit(self, users, items, labels, n_negatives):
        labels_ = labels.flatten()
        labels_[labels_ == 0] = -1  # dislike
        users_ = np.repeat(users, n_negatives + 1)
        items_ = items.flatten()

        self.matrix = csr_matrix(
            (labels_, (users_, items_)), shape=(len(users_), len(items_))
        )
        self.model.fit(self.matrix)
        self.is_train = True

    def pred_top_k(self, user_id, K: int = 30):
        if not self.is_train:
            raise ValueError("Model should be trained")

        ids, scores = self.model.recommend(
            user_id,
            self.matrix[user_id],
            N=K,
            filter_already_liked_items=False,
        )
        res = np.array([scores, ids])
        return torch.tensor(res)


def collate_fn(batch, num_negatives, num_items):
    users, target_items, users_negatives = [], [], []

    for triplets in batch:
        user, target_item, seen_item = triplets

        users.append(user)
        target_items.append(target_item)
        user_negatives = []

        while len(user_negatives) < num_negatives:
            candidate = random.randint(0, num_items)
            if candidate not in seen_item:
                user_negatives.append(candidate)

        users_negatives.append(user_negatives)

    positive = torch.ones(len(batch), 1)
    negatives = torch.zeros(len(batch), num_negatives)
    labels = torch.hstack([positive, negatives])
    items = torch.hstack(
        [torch.tensor(target_items).reshape(-1, 1), torch.tensor(users_negatives)]
    )
    return torch.hstack(users), items, labels


def calculate_hitrate(df_preds: pd.DataFrame, K: int):
    return df_preds[df_preds["rank"] < K].groupby("user_index")["relevant"].max().mean()


def calculate_precision(df_preds: pd.DataFrame, K: int):
    return (
        df_preds[df_preds["rank"] < K].groupby("user_index")["relevant"].mean()
    ).mean()


def plot(metrics: dict[str, list[float]]):
    fig, (loss, hitrate, precision) = plt.subplots(1, 3, figsize=[18, 5])

    loss.plot(
        metrics["losses"],
        label="Loss",
        c="black",
    )
    hitrate.plot(
        metrics["hitrate"],
        label="Hitrate",
        c="black",
    )
    precision.plot(
        metrics["precision"],
        label="Precision",
        c="black",
    )

    loss.legend()
    hitrate.legend()
    precision.legend()
    return fig


def get_preds(model, df: pd.DataFrame, run_params):
    users = df["user_index"].unique()
    preds = model.pred_top_k(torch.tensor(users), run_params["K"])[1].numpy()
    df_preds = pd.DataFrame(
        {
            "node_index": list(preds),
            "user_index": users,
            "rank": [[j for j in range(0, run_params["K"])] for i in range(len(preds))],
        }
    )

    df_preds = df_preds.explode(["node_index", "rank"]).merge(
        df[["user_index", "node_index"]].assign(relevant=1).drop_duplicates(),
        on=["user_index", "node_index"],
        how="left",
    )
    df_preds["relevant"] = df_preds["relevant"].fillna(0)

    return df_preds


def read_node2name() -> dict:
    with open(NOTEBOOK_PARAMS["node2name"], mode="r", encoding="utf-8") as file:
        node2name = json.load(file)
    node2name = {int(k): v for k, v in node2name.items()}
    return node2name


def prepare_data() -> tuple:
    node2name = read_node2name()
    df = pd.read_parquet(NOTEBOOK_PARAMS["clickstream"])
    df = df.head(NOTEBOOK_PARAMS["df_limit"])
    df["is_train"] = df["event_date"] < df["event_date"].max() - pd.Timedelta("2 day")

    df["names"] = df["node_id"].map(node2name)

    train_cooks = df[df["is_train"]]["cookie_id"].unique()
    train_items = df[df["is_train"]]["node_id"].unique()

    df = df[(df["cookie_id"].isin(train_cooks)) & (df["node_id"].isin(train_items))]

    user_indes, index2user_id = pd.factorize(df["cookie_id"])
    df.loc[:, "user_index"] = user_indes

    node_indes, index2node = pd.factorize(df["node_id"])
    df.loc[:, "node_index"] = node_indes

    df_train, df_val = df[df["is_train"]], df[~df["is_train"]]
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    print(f"Train shape: {df_train.shape}\n" f"Val   shape: {df_val.shape}")

    user2seen = df_train.groupby("user_index")["node_index"].agg(lambda x: list(set(x)))

    print(df_train.sample(5))

    return (
        df_train,
        df_val,
        node2name,
        user2seen,
        user_indes,
        index2user_id,
        node_indes,
        index2node,
    )


def create_loader(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    user2seen: pd.DataFrame,
    run_params: dict,
):
    train_dataset = RecDataset(
        df_train["user_index"].values,
        df_train["node_index"].values,
        user2seen,
    )

    dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=run_params["n_workers"],
        batch_size=run_params["batch_size"],
        collate_fn=lambda batch: collate_fn(
            batch=batch,
            num_negatives=run_params["n_negatives"],
            num_items=max(
                df_train["node_index"].max(),
                df_val["node_index"].max(),
            ),
        ),
    )

    return dataloader
