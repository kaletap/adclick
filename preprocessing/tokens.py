import tensorflow as tf


class Vocabulary:
    def __init__(self, vocab_file: str, vocab_size: int):
        """
        :param vocab_file: file with vocabulary ordered by popularity of a word
        :param vocab_size: maximum size of a vocabulary (we use top vocab_size words of vocab_file)
        """
        words = ["<PAD>", "<UNK>"]  # pad and unknown token, represented with 0 and 1 respectively
        with open(vocab_file) as f:
            try:
                for i in range(vocab_size):
                    words.append(next(f).strip())
            except StopIteration:  # TODO: should I maybe write it better?
                raise StopIteration("Vocabulary file contains only {} lines".format(i))
        self.word2idx = {word: num for num, word in enumerate(words)}
        self.idx2word = words

        table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant([0, 1, 2, 3]),
                values=tf.constant([10, 11, 12, 13]),
            ),
            default_value=tf.constant(-1),
            name="class_weight"
        )

    def get_word_num(self, word):
        print(word.dtype)
        print(word.numpy())
        return self.word2idx.get(word.numpy(), 1)

    def __getitem__(self, word):
        if isinstance(word, tf.Tensor):
            return tf.py_function(func=self.get_word_num, inp=[word], Tout=tf.int32, name="word2idx")
        else:
            return self.word2idx[word]

    def __call__(self, word):
        return self[word]


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
