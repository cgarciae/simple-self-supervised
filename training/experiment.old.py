import typing
from pathlib import Path
from copy import copy

import dataget
import dicto
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skorch

# import tensorflow as tf
import torch
import typer
from imblearn.over_sampling import RandomOverSampler
from plotly import express as px
from plotly import graph_objs as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skorch.helper import predefined_split


HAS_VIZ = False


def main(
    params_path: Path = "training/params.yml",
    viz: bool = False,
    toy: bool = False,
    model: str = "torch",
) -> None:

    torch.autograd.set_detect_anomaly(True)

    params = dicto.load(params_path)

    df_train, df_test = dataget.toy.spirals().get()

    X_train = df_train[["x0", "x1"]].to_numpy()
    y_train = df_train["y"].to_numpy()
    X_test = df_test[["x0", "x1"]].to_numpy()
    y_test = df_test["y"].to_numpy()

    transform = MinMaxScaler()
    X_train = transform.fit_transform(X_train)
    X_test = transform.transform(X_test)

    _ds_train = ContrastiveDataset(
        X_train,
        y_train,
        batch_size=params.batch_size,
        steps_per_epoch=params.steps_per_epoch,
        noise_std=params.noise_std,
        n_neighbors=params.n_neighbors,
        n_hops=params.n_hops,
        # transform=torch.tensor,
        viz=viz,
    )

    if viz:
        ds = copy(_ds_train)

        x, labels = next(iter(ds))

        x = x.reshape(3, -1, x.shape[-1])

        batch_size = len(x) // 3

        anchor = x[0] + np.random.normal(scale=0.01, size=x[0].shape)
        positive = x[1] + np.random.normal(scale=0.01, size=x[0].shape)
        negative = x[2] + np.random.normal(scale=0.01, size=x[0].shape)

        edge_x = []
        edge_y = []
        for i in range(len(anchor)):
            edge_x.append(positive[i, 0])
            edge_x.append(anchor[i, 0])
            edge_x.append(negative[i, 0])
            edge_x.append(None)

            edge_y.append(positive[i, 1])
            edge_y.append(anchor[i, 1])
            edge_y.append(negative[i, 1])
            edge_y.append(None)

        fig = go.Figure(
            data=[
                go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none",),
                go.Scatter(
                    x=anchor[:, 0],
                    y=anchor[:, 1],
                    mode="markers",
                    name="anchor",
                    marker=dict(color="black"),
                ),
                go.Scatter(
                    x=positive[:, 0],
                    y=positive[:, 1],
                    mode="markers",
                    name="positive",
                    marker=dict(color="yellow"),
                ),
                go.Scatter(
                    x=negative[:, 0],
                    y=negative[:, 1],
                    mode="markers",
                    name="negative",
                    marker=dict(color="red"),
                ),
            ]
        )

        fig.show()

    # tensorflow
    if model == "tf":
        ds_train = tf.data.Dataset.from_generator(
            lambda: _ds_train, (tf.float32, tf.float32), ([None, 2], [None, 1]),
        )

        ds_test = ContrastiveDataset(
            X_test,
            y_test,
            batch_size=params.batch_size,
            steps_per_epoch=params.steps_per_epoch,
            noise_std=params.noise_std,
            n_neighbors=params.n_neighbors,
            n_hops=params.n_hops,
            # transform=torch.tensor,
            viz=viz,
        )
        model = SiameseNetwork(n_layers=params.n_layers, n_units=params.n_units)

        model.compile(
            loss=tf.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(lr=params.lr),
            metrics=[tf.metrics.BinaryAccuracy()],
        )
        model(X_train[: 3 * 2])
        model.summary()

        model.fit(ds_train, epochs=params.epochs)

        if viz:
            h = model(X_train, return_embeddings=True)

            px.scatter(x=X_train[:, 0], y=X_train[:, 1], color=h[:, 0]).show()
    else:
        # pytorch
        model = ContrastiveNet(
            batch_size=params.batch_size * 2,
            n_layers=params.n_layers,
            n_units=params.n_units,
            embedding_size=params.embedding_size,
        )

        net = skorch.NeuralNet(
            model,
            criterion=criterion,
            batch_size=None,
            max_epochs=params.epochs,
            lr=params.lr,
            optimizer=torch.optim.Adam,
            # train_split=predefined_split(ds_test),
            train_split=None,
            device="cuda",
        )

        net.fit(_ds_train, y=None)

        if viz:
            net.module.eval()
            h = (
                net.module(
                    torch.tensor(X_train, dtype=torch.float32, device="cuda"),
                    return_embeddings=True,
                )
                .cpu()
                .detach()
                .numpy()
            )

            px.scatter(x=X_train[:, 0], y=X_train[:, 1], color=h[:, 0]).show()


def criterion(**kwargs):
    def get_loss(y_pred, y_true):

        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)

        # y_pred = y_pred.reshape(-1, 3, y_pred.shape[-1])
        # loss = torch.nn.functional.triplet_margin_loss(anchor, positive, negative)

        return loss

    return get_loss


class ContrastiveNet(torch.nn.Module):
    def __init__(
        self, batch_size: int, n_layers: int, n_units: int, embedding_size: int
    ):
        super().__init__()

        mask = self.get_mask(batch_size)

        self.register_buffer("mask", torch.tensor(mask))

        f_list = []

        for i in range(n_layers):
            f_list += [
                torch.nn.Linear(n_units if i > 0 else 2, n_units),
                torch.nn.BatchNorm1d(n_units),
                torch.nn.ReLU(),
            ]

        f_list += [
            torch.nn.Linear(n_units, embedding_size),
        ]

        self.f = torch.nn.Sequential(*f_list)

        self.discriminator = torch.nn.Linear(embedding_size, 1)

    def forward(self, x, return_embeddings=False):

        z = self.f(x)

        if return_embeddings:
            return z

        # z = self.g(z)

        # z = z / torch.norm(z, p=2, dim=1, keepdim=True)
        # z = torch.mm(z, z.t()) + self.mask

        z = z.reshape(3, -1, z.shape[-1])

        anchor = z[0]
        positive = z[1]
        negative = z[2]

        z = torch.cat(
            [torch.abs(anchor - positive), torch.abs(anchor - negative)], dim=0
        )

        z = self.discriminator(z)

        return z

    def get_mask(self, batch_size):

        mask = np.zeros((batch_size, batch_size), dtype=np.float32)
        np.fill_diagonal(mask, -3.4028235e38)

        return mask


class ContrastiveDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        x,
        y,
        batch_size: int,
        steps_per_epoch: int,
        noise_std: float,
        n_neighbors: int,
        n_hops: int,
        transform: typing.Callable = None,
        viz: bool = False,
    ):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.noise_std = noise_std
        self.n_neighbors = n_neighbors
        self.n_hops = n_hops
        self.transform = transform
        self.viz = viz
        self.labels = np.concatenate(
            [np.ones((self.batch_size, 1)), np.zeros((self.batch_size, 1))], axis=0
        ).astype(np.float32)

        N = len(self.x)

        nbrs = NearestNeighbors(
            n_neighbors=self.n_neighbors, algorithm="ball_tree"
        ).fit(self.x)
        distances, neighbors = nbrs.kneighbors(self.x)

        self.relative_neighbors = []

        for i in range(N):
            node_neighbors = set(neighbors[i])

            for _ in range(self.n_hops):
                new_neighbors = neighbors[list(node_neighbors)].reshape(-1)
                node_neighbors |= set(new_neighbors)

            self.relative_neighbors.append(list(node_neighbors))

        if self.viz:
            plot_graph(self.x, self.relative_neighbors)

    def __iter__(self):

        N = len(self.x)

        for step in range(self.steps_per_epoch):

            idx_anchor = np.random.choice(N, self.batch_size, replace=True)
            idx_positive = [
                self.get_positive_for(self.relative_neighbors[i], i) for i in idx_anchor
            ]
            idx_negegative = [
                self.get_negative_for(self.relative_neighbors[i], N) for i in idx_anchor
            ]

            anchor = self.x[idx_anchor]
            pos = self.x[idx_positive]
            neg = self.x[idx_negegative]

            x = (
                np.concatenate([anchor, pos, neg], axis=0)
                .reshape(-1, 2)
                .astype(np.float32)
            )

            # x = x + np.random.normal(scale=self.noise_std, size=x.shape)

            x = x.astype(np.float32)

            if self.transform:
                x = self.transform(x)

            yield x, self.labels

    def get_negative_for(self, node_neighbors, N):

        while True:
            negative_index = np.random.randint(N)

            if negative_index not in node_neighbors:
                return negative_index

    def get_positive_for(self, node_neighbors, i):
        N = len(node_neighbors)

        while True:
            positive_index = np.random.randint(N)
            positive_index = node_neighbors[positive_index]

            if positive_index != i:
                return positive_index


def plot_graph(X, neighbors):
    N = len(X)

    edge_x = []
    edge_y = []
    for i in range(N):
        for j in neighbors[i]:
            edge_x.append(X[i, 0])
            edge_x.append(X[j, 0])
            edge_x.append(None)

            edge_y.append(X[i, 1])
            edge_y.append(X[j, 1])
            edge_y.append(None)

    go.Figure(
        [
            go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none",),
            go.Scatter(
                x=X[:, 0], y=X[:, 1], mode="markers", marker=dict(color="black")
            ),
        ]
    ).show()


# class SiameseNetwork(tf.keras.Model):
#     def __init__(
#         self, n_layers=1, n_units=16,
#     ):
#         super().__init__()

#         self.model = tf.keras.Sequential(
#             [tf.keras.layers.Dense(n_units, activation="relu") for _ in range(n_layers)]
#             + [tf.keras.layers.Dense(1)]
#         )

#         self.discriminator = tf.keras.layers.Dense(1, activation="sigmoid")

#     def call(self, inputs, return_embeddings=False):

#         if return_embeddings:
#             return self.model(inputs)

#         inputs = tf.reshape(inputs, [-1, 3, inputs.shape[-1]])

#         anchor = inputs[:, 0]
#         positive = inputs[:, 1]
#         negative = inputs[:, 2]

#         x1 = tf.concat([anchor, anchor], axis=0)
#         x2 = tf.concat([positive, negative], axis=0)

#         embedding1 = self.model(x1)
#         embedding2 = self.model(x2)

#         net = tf.abs(embedding1 - embedding2)

#         net = self.discriminator(net)

#         return net


# class PairwiseRankingLoss(tf.losses.Loss):
#     def __init__(self, margin=1):
#         super().__init__()
#         self.margin = margin

#     def call(self, y_true, y_pred):
#         norm = y_pred
#         y_true = tf.cast(y_true, tf.float32)

#         return (
#             y_true * norm
#         )  # + (1 - y_true) * tf.math.maximum(0.0, self.margin - norm)


# class MeanPairwiseRankingLoss(tf.keras.metrics.Mean):
#     def __init__(self, margin=1, **kwargs):
#         super().__init__(**kwargs)
#         self.margin = margin

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         norm = y_pred
#         y_true = tf.cast(y_true, tf.float32)

#         loss = (
#             y_true * norm
#         )  # + (1 - y_true) * tf.math.maximum(0.0, self.margin - norm)

#         return super().update_state(loss, sample_weight=sample_weight)


def decision_boundaries(X_in, model, n=10):

    X = X_in[0]
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    hx = (x_max - x_min) / n
    hy = (y_max - y_min) / n
    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

    # Obtain labels for each point in mesh using the model.
    points = np.c_[xx.ravel(), yy.ravel()]
    X_pred = np.tile(X_in, (len(points), 1, 1))

    points = points[:, np.newaxis, :]
    print("points", points.shape)
    print("X_pred", X_pred.shape)

    points = np.concatenate([X_pred, points], axis=1)

    Z = model.predict(points)
    Z = Z[:, -1, :]
    Z = np.argmax(Z, axis=-1)

    zz = Z.reshape(xx.shape)

    return xx, yy, zz


if __name__ == "__main__":
    typer.run(main)
