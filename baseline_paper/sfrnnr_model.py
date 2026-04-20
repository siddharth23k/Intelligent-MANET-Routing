
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

_serializable = tf.keras.utils.register_keras_serializable

# Same nine factors as the legacy weighted-LFP baseline
FACTOR_COLS = [
    "d_res",
    "LET",
    "ND",
    "RSSI",
    "LS",
    "LA",
    "LQ_mean",
    "LL_d",
    "T_hello",
]

N_FACTORS = len(FACTOR_COLS)
# Default membership count
N_MFS = 2


@_serializable(package="SFRNNR")
class FuzzificationLayer(layers.Layer):
    """Gaussian fuzzification: each scalar maps to N_MFS membership degrees."""

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
        c = tf.reshape(self.centers, (1, 1, self.n_inputs, self.n_mfs))
        w = tf.exp(self.log_widths) + 1e-4
        w = tf.reshape(w, (1, 1, self.n_inputs, self.n_mfs))
        mu = tf.exp(-tf.square(x - c) / (2.0 * tf.square(w)))
        b, t, _, _ = tf.unstack(tf.shape(mu))
        return tf.reshape(mu, (b, t, self.out_dim))

    def get_config(self):
        base = super().get_config()
        base.update({"n_inputs": self.n_inputs, "n_mfs": self.n_mfs})
        return base


@_serializable(package="SFRNNR")
class ThresholdScale(layers.Layer):
    """Maps sigmoid output to [0.2, 0.8]."""

    def call(self, inputs):
        return 0.2 + 0.6 * inputs

    def get_config(self):
        return super().get_config()


def build_sfrnnr_model(
    seq_len: int | None = None,
    n_factors: int = N_FACTORS,
    n_mfs: int = N_MFS,
    gru_units: int = 8,
    rule_units: int = 4,
    thr_hidden: int = 4,
    dropout: float = 0.0,
    name: str = "SFRNNR",
) -> keras.Model:
    in_shape = (seq_len, n_factors) if seq_len is not None else (None, n_factors)
    inputs = keras.Input(shape=in_shape, name="factor_sequence")
    fuzz = FuzzificationLayer(n_inputs=n_factors, n_mfs=n_mfs, name="fuzzification")(inputs)
    rnn = layers.GRU(
        gru_units,
        return_sequences=True,
        activation="tanh",
        name="fuzzy_rnn",
    )(fuzz)
    if dropout and dropout > 0:
        rnn = layers.Dropout(dropout, name="fuzzy_rnn_dropout")(rnn)
    norm = layers.LayerNormalization(name="normalization")(rnn)
    cons = layers.Dense(rule_units, activation="relu", name="consequent")(norm)
    lfp = layers.Dense(1, activation="sigmoid", name="summation_lfp")(cons)
    thr_h = layers.Dense(max(1, thr_hidden), activation="relu", name="threshold_hidden")(norm)
    thr_log = layers.Dense(1, activation="sigmoid", name="threshold_logit")(thr_h)
    thr = ThresholdScale(name="output_threshold")(thr_log)

    model = keras.Model(
        inputs=inputs,
        outputs={"lfp": lfp, "lfp_threshold": thr},
        name=name,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(2e-3),
        loss={
            "lfp": keras.losses.BinaryCrossentropy(),
            "lfp_threshold": keras.losses.MeanSquaredError(),
        },
        loss_weights={"lfp": 1.0, "lfp_threshold": 0.25},
    )
    return model


def custom_objects_dict():
    return {
        "FuzzificationLayer": FuzzificationLayer,
        "ThresholdScale": ThresholdScale,
    }


def load_sfrnnr(path: str, compile_model: bool = False) -> keras.Model:
    return keras.models.load_model(
        path, custom_objects=custom_objects_dict(), compile=compile_model
    )
