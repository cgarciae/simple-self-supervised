from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import typer
from plotly import express as px
from plotly import graph_objs as go
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import RandomOverSampler


def main(
    data_path: Path = typer.Option(...),
    n_classes: int = typer.Option(...),
    memory_size: int = 8,
    n_layers: int = 3,
    n_neurons: int = 16,
    radius=1.5,
    epochs: int = 400,
    viz: bool = False,
) -> None:

    train_path = data_path / "training-set.csv"
    # test_path = data_path / "test-set.csv"

    df_train = pd.read_csv(train_path, names=["x1", "x2", "label"])
    # df_test = pd.read_csv(test_path, names=["x1", "x2", "label"])

    X = df_train[["x1", "x2"]].to_numpy()
    # y_train = df_train["label"].to_numpy().astype(int)

    # X_test = df_test[["x1", "x2"]].to_numpy()
    # y_test = df_test["label"].to_numpy().astype(int)

    X, y = preprocess(X, radius)

    print(X.shape, y.shape, y.sum())

    return

    # transform = StandardScaler()
    # X_train = transform.fit_transform(X_train)
    # X_test = transform.transform(X_test)

    model = MyModel(
        memory_size=memory_size,
        memory_output_size=2,
        n_labels=n_classes,
        n_heads=n_heads,
        n_layers=n_layers,
        n_neurons=n_neurons,
    )
    model.predict(X_train)
    model.summary()

    model.compile(
        optimizer=tf.optimizers.Adam(0.001),
        loss=tf.losses.sparse_categorical_crossentropy,
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        x=X_train,
        y=y_train,
        batch_size=1,
        epochs=epochs,
        validation_data=(X_test, y_test),
    )
    model.summary()

    if viz:
        fig = px.scatter(None, X_train[0, :, 0], X_train[0, :, 1], color=y_train[0])

        xx, yy, zz = decision_boundaries(X_train, model)
        xx = xx[0]
        yy = yy[:, 0]

        fig.add_trace(go.Contour(x=xx, y=yy, z=zz, opacity=0.5))

        fig.show()


class MyModel(tf.keras.Model):
    def __init__(
        self, n_labels, n_layers=1, n_neurons=16,
    ):
        super().__init__()

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(n_neurons, activation="relu")
                for _ in range(n_layers)
            ]
            + [tf.keras.layers.Dense(1)]
        )

        self.mhsa_modules = [
            tf.keras.Sequential(
                [
                    MultiHeadSelfAttention(n_neurons, n_heads),
                    tf.keras.layers.Dense(n_neurons, activation="relu"),
                ]
            )
            for _ in range(n_layers)
        ]

        self.dense_out = tf.keras.layers.Dense(n_labels, activation="softmax")

    def call(self, inputs):

        net = inputs
        print(net.shape)

        self.embeddings(net)
        print(net.shape)

        for module in self.mhsa_modules:
            net = module(inputs)
            print("msha", net.shape)

        # net = net + net0

        net = self.dense_out(net)
        print(net.shape)

        return net


def preprocess(X, radius):
    N = len(X)
    X0 = X[np.newaxis, ...]
    X1 = X[:, np.newaxis, ...]

    distance = np.linalg.norm(X0 - X1, axis=-1)

    y = (distance <= radius).astype(np.int32)

    # print("Neighbors", y.sum(axis=1) - 1)
    # is_close = is_close[..., np.newaxis]

    X0 = np.tile(X0, (N, 1, 1))
    X1 = np.tile(X1, (1, N, 1))

    X = np.concatenate([X0, X1], axis=-1)

    X = X.reshape((-1, 4))
    y = y.reshape((-1,))

    X, y = RandomOverSampler(random_state=42).fit_resample(X, y)

    return X, y


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
