"""SFRNNR: our reimplementation of the base paper's fuzzy recurrent predictor.

Built from the published description, since no code was released: fuzzify the
nine link factors, encode with a GRU, normalise, apply a consequent layer,
defuzzify into a link failure probability, and emit an adaptive threshold from a
second head.

Scoping note: this is fuzzy inspired, not a faithful ANFIS. Membership functions
are trainable Gaussians, but there is no explicit rule layer computing a T norm
across inputs; the consequent dense layer learns an arbitrary combination. That
buys end to end differentiability and costs rule level interpretability.
"""

from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from bootstrap import setup_paths  # noqa: E402

setup_paths()

from schema import FACTOR_COLS  # noqa: E402

_serializable = tf.keras.utils.register_keras_serializable

N_FACTORS = len(FACTOR_COLS)
N_MFS = 2


@_serializable(package="SFRNNR")
class FuzzificationLayer(layers.Layer):
    """Gaussian membership degrees, n_mfs per input factor.

    Widths are stored as logs and exponentiated, so they stay positive without a
    constraint or a clip.
    """

    def __init__(self, n_inputs: int = N_FACTORS, n_mfs: int = N_MFS, **kwargs):
        super().__init__(**kwargs)
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.out_dim = n_inputs * n_mfs

    def build(self, input_shape):
        self.centers = self.add_weight(
            name="mf_centers",
            shape=(self.n_inputs, self.n_mfs),
            initializer=keras.initializers.RandomUniform(0.15, 0.85),
            trainable=True,
        )
        self.log_widths = self.add_weight(
            name="mf_log_widths",
            shape=(self.n_inputs, self.n_mfs),
            initializer=keras.initializers.Constant(-1.0),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        x = tf.expand_dims(inputs, -1)
        centers = tf.reshape(self.centers, (1, 1, self.n_inputs, self.n_mfs))
        widths = tf.reshape(tf.exp(self.log_widths) + 1e-4, (1, 1, self.n_inputs, self.n_mfs))
        membership = tf.exp(-tf.square(x - centers) / (2.0 * tf.square(widths)))
        batch, steps = tf.shape(membership)[0], tf.shape(membership)[1]
        return tf.reshape(membership, (batch, steps, self.out_dim))

    def get_config(self):
        config = super().get_config()
        config.update({"n_inputs": self.n_inputs, "n_mfs": self.n_mfs})
        return config


@_serializable(package="SFRNNR")
class ThresholdScale(layers.Layer):
    """Squash the threshold head into the paper's [0.2, 0.8] band."""

    def call(self, inputs):
        return 0.2 + 0.6 * inputs

    def get_config(self):
        return super().get_config()


def build_sfrnnr_model(
    seq_len: int | None = None,
    n_factors: int = N_FACTORS,
    n_mfs: int = N_MFS,
    gru_units: int = 16,
    rule_units: int = 8,
    thr_hidden: int = 8,
    dropout: float = 0.0,
    learning_rate: float = 2e-3,
    run_eagerly: bool = False,
    name: str = "SFRNNR",
) -> keras.Model:
    """Build the SFRNNR.

    run_eagerly skips graph compilation.

    Keras 3 retraces its train step whenever the batch shape changes, and the
    final batch of an epoch is usually a different size from the rest. On a
    small run the tracing cost dominates completely, and on some CPU builds it
    stalls outright. Real training amortises a handful of traces over many
    epochs, so graph mode stays the default there; the smoke configuration runs
    eagerly. pipeline/diagnose_tf.py measures this directly.
    """
    input_shape = (seq_len, n_factors) if seq_len is not None else (None, n_factors)
    inputs = keras.Input(shape=input_shape, name="factor_sequence")

    fuzzified = FuzzificationLayer(n_inputs=n_factors, n_mfs=n_mfs, name="fuzzification")(inputs)
    encoded = layers.GRU(gru_units, return_sequences=True, name="fuzzy_rnn")(fuzzified)
    if dropout > 0:
        encoded = layers.Dropout(dropout, name="fuzzy_rnn_dropout")(encoded)
    normalised = layers.LayerNormalization(name="normalization")(encoded)

    consequent = layers.Dense(rule_units, activation="relu", name="consequent")(normalised)
    lfp = layers.Dense(1, activation="sigmoid", name="summation_lfp")(consequent)

    threshold_hidden = layers.Dense(max(1, thr_hidden), activation="relu", name="threshold_hidden")(normalised)
    threshold_logit = layers.Dense(1, activation="sigmoid", name="threshold_logit")(threshold_hidden)
    threshold = ThresholdScale(name="output_threshold")(threshold_logit)

    model = keras.Model(
        inputs=inputs, outputs={"lfp": lfp, "lfp_threshold": threshold}, name=name
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss={
            "lfp": keras.losses.BinaryCrossentropy(),
            "lfp_threshold": keras.losses.MeanSquaredError(),
        },
        loss_weights={"lfp": 1.0, "lfp_threshold": 0.25},
        # String form on purpose: a metric instance on a multi output model
        # triggers repeated retracing on some Keras 3 builds.
        metrics={"lfp": "auc"},
        run_eagerly=bool(run_eagerly),
        jit_compile=False,
    )
    return model


def custom_objects_dict() -> dict:
    return {"FuzzificationLayer": FuzzificationLayer, "ThresholdScale": ThresholdScale}


def load_sfrnnr(path: str, compile_model: bool = False) -> keras.Model:
    return keras.models.load_model(path, custom_objects=custom_objects_dict(), compile=compile_model)
