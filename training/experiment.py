from pathlib import Path

import pandas as pd
import tensorflow as tf
import typer
from plotly import express as px
from plotly import graph_objs as go
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


def main(
    data_path: Path = typer.Option(...),
    n_classes: int = typer.Option(...),
    memory_size: int = 8,
    n_layers: int = 0,
    n_neurons: int = 16,
    k: float = 1.0,
    epochs: int = 400,
    viz: bool = False,
) -> None:

    train_path = data_path / "training-set.csv"
    test_path = data_path / "test-set.csv"

    df_train = pd.read_csv(train_path, names=["x1", "x2", "label"])
    df_test = pd.read_csv(test_path, names=["x1", "x2", "label"])

    X_train = df_train[["x1", "x2"]].to_numpy()
    y_train = df_train["label"].to_numpy().astype(int)

    X_test = df_test[["x1", "x2"]].to_numpy()
    y_test = df_test["label"].to_numpy().astype(int)

    transform = StandardScaler()

    X_train = transform.fit_transform(X_train)
    X_test = transform.transform(X_test)

    model = MyModel(
        (2,),
        memory_size=memory_size,
        n_labels=n_classes,
        k=k,
        n_layers=n_layers,
        n_neurons=n_neurons,
    )

    model.summary()

    model.compile(
        optimizer=tf.optimizers.Adam(0.001),
        loss=tf.losses.sparse_categorical_crossentropy,
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        x=X_train,
        y=y_train,
        batch_size=32,
        epochs=epochs,
        validation_data=(X_test, y_test),
    )

    simple_memory = model.layers[1]

    if viz:
        fig = px.scatter(None, X_train[:, 0], X_train[:, 1], color=y_train)

        memory = simple_memory.memory.kernel.numpy()

        xx, yy, zz = decision_boundaries(X_train, model)
        xx = xx[0]
        yy = yy[:, 0]

        fig.add_trace(go.Contour(x=xx, y=yy, z=zz, opacity=0.5))
        fig.add_trace(
            go.Scatter(
                x=memory[:, 0],
                y=memory[:, 1],
                mode="markers",
                marker=dict(color="black", size=10),
            )
        )

        fig.show()


def MyModel(input_shape, memory_size, n_labels, k=1, n_layers=0, n_neurons=16):

    inputs = tf.keras.Input(shape=input_shape)

    net = SimpleMemory(memory_size, k=k)(inputs)

    for i in range(n_layers):
        net = tf.keras.layers.Dense(n_neurons, activation="relu")(net)

    net = tf.keras.layers.Dense(n_labels, activation="softmax")(net)

    model = tf.keras.Model(inputs=inputs, outputs=net)

    return model


class SimpleMemory(tf.keras.Model):
    def __init__(self, memory_size, k=1):
        super().__init__()
        self.memory_size = memory_size
        self.k = k

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(self.memory_size)
        self.memory = tf.keras.layers.Dense(input_shape[-1], use_bias=False)

    def call(self, inputs):

        net = self.dense(inputs)
        net = tf.nn.softmax(net * self.k)
        net = self.memory(net)

        return net


def decision_boundaries(X, model, n=10):

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    hx = (x_max - x_min) / n
    hy = (y_max - y_min) / n
    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

    # Obtain labels for each point in mesh using the model.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    preds = model.predict(np.c_[xx.ravel(), yy.ravel()])

    preds = np.argmax(preds, axis=-1)

    zz = preds.reshape(xx.shape)

    return xx, yy, zz


if __name__ == "__main__":
    typer.run(main)
