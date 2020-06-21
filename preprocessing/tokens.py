import tensorflow as tf


class Vocabulary:
    def __init__(self, vocab_file: str, vocab_size: int):
        """
        :param vocab_file: file with vocabulary ordered by popularity of a word
        :param vocab_size: maximum size of a vocabulary (we use top vocab_size words of vocab_file)
        """
        words = ["<PAD>", "<UNK>"]  # pad and unknown token, represented with 0 and 1 respectively
        with open(vocab_file) as f:
            for _ in range(vocab_size):
                words.append(next(f).strip())
        self.word2idx = {word: num for num, word in enumerate(words)}
        self.idx2word = words

    def __getitem__(self, word):
        print(tf.executing_eagerly())
        if isinstance(word, tf.Tensor):
            word = word.numpy()
        return self.word2idx[word]

    def __call__(self, word):
        return self[word]


# Sequence of tokens transformer
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
