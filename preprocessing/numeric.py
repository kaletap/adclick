import pandas as pd
import tensorflow as tf


raw_stats = pd.read_csv("data/stats.csv")
stats = raw_stats.T
stats.columns = raw_stats["column"]
stats = stats.drop("column")


def get_stats(column):
    rows = stats.loc[stats["column"] == column]
    return rows["mean"].values[0], rows["sd"].values[0]


# Label transformer
class ExtractLabels:
    def __call__(self, features, labels):
        return labels


# Numeric variables transformers
class PackNumericFeatures:
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features[name] for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features["numeric"] = numeric_features
        return features, labels


class Normalize:
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        data = features["numeric"]
        mean = stats.loc["mean", self.names]
        std = stats.loc["sd", self.names]
        features["numeric"] = (data - mean) / std
        return features, labels


class Categorize:
    def __init__(self, name, size):
        self.name = name
        self.size = size

    def __call__(self, features, labels):
        features[self.name] = tf.one_hot(features[self.name], self.size)
        return features, labels


class FeatureHash:
    def __init__(self, name: str, n_buckets: int = 1000):
        self.name = name
        self.n_buckets = n_buckets

    def __call__(self, features, labels):
        ids = tf.as_string(features[self.name])
        ids = tf.strings.to_hash_bucket_fast(ids, self.n_buckets)
        features[self.name + "_hashed"] = ids
        return features, labels


class FeatureHashMany:
    def __init__(self, names: str, n_buckets: int = 1000):
        self.names = names
        self.n_buckets = n_buckets

    def __call__(self, features, labels):
        for name in self.names:
            ids = tf.as_string(features[name])
            ids = tf.strings.to_hash_bucket_fast(ids, self.n_buckets)
            features[name + "_hashed"] = ids
        return features, labels
