import tensorflow as tf


# Sequence of tokens transformer
class TokenizeWithVocabFile:
    def __init__(self, name: str, vocab_size: int, max_length: int, vocab_file: str, oov_buckets: int = 10):
        self.name = name
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.table = tf.lookup.StaticVocabularyTable(
            tf.lookup.TextFileInitializer(vocab_file,
                                          tf.string,
                                          tf.lookup.TextFileIndex.WHOLE_LINE,
                                          tf.int64,
                                          tf.lookup.TextFileIndex.LINE_NUMBER,
                                          delimiter=" ",
                                          vocab_size=vocab_size),
            num_oov_buckets=oov_buckets
        )

    def __call__(self, features, labels):
        tokens_list = features[self.name]
        tokens_list = tf.strings.split(tokens_list, sep="|").to_tensor(default_value="<PAD>", shape=[None, self.max_length])
        tokens = self.table.lookup(tokens_list)
        # tokens = tf.strings.to_hash_bucket_fast(tokens_list, num_buckets=self.vocab_size)
        features[self.name] = tokens
        return features, labels


class Tokenize:
    def __init__(self, name: str, vocab_size: int, max_length: int):
        self.name = name
        self.vocab_size = vocab_size
        self.max_length = max_length
        # self.vocab = Vocabulary(vocab_file, vocab_size)

    def __call__(self, features, labels):
        tokens_list = features[self.name]
        tokens_list = tf.strings.split(tokens_list, sep="|").to_tensor(default_value="<PAD>", shape=[None, self.max_length])
        # hashing instead of proper vocabulary
        tokens = tf.strings.to_hash_bucket_fast(tokens_list, num_buckets=self.vocab_size)
        features[self.name] = tokens
        return features, labels
