import pandas as pd
import tensorflow as tf


raw_stats = pd.read_csv("data/stats.csv")
stats = raw_stats.T
stats.columns = raw_stats["column"]
stats = stats.drop("column")


def get_stats(column):
    rows = stats.loc[stats["column"] == column]
    return rows["mean"].values[0], rows["sd"].values[0]


class PackNumericFeatures:
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features, labels


def get_normalize_function(numeric_features):
    def normalize(data):
        mean = stats.loc["mean", numeric_features]
        std = stats.loc["sd", numeric_features]
        return (data - mean) / std
    return normalize


#
# numeric_layer = tf.keras.layers.DenseFeatures(feature_columns)
# embedding_layer = tf.keras.layers.DenseFeatures(embedding_columns)
#
# x = tf.keras.Input(shape=4)
# x_num = numeric_layer(x)
# x_num = tf.keras.layers.Dense(512, activation="relu")
#
# x_emb = embedding_layer(x)
# x = tf.keras.layers.Concatenate([x_num, x_emb])
# x = tf.keras.layers.Dense(256, activation="relu")
# x = tf.keras.layers.Dense(1, activation="sigmoid")
#
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(lr=lr),
#     loss=tf.keras.losses.BinaryCrossentropy(),
#     metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
# )
# history = model.fit(packed_train, validation_data=packed_test, epochs=10, callbacks=[scheduler])