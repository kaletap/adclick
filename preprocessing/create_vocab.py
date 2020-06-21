import argparse
import csv
import json
from collections import Counter, OrderedDict
from tqdm import tqdm


"""
Script to extract a vocabulary from columns containing sequence of tokens.
From a data description (https://www.kaggle.com/c/kddcup2012-track2/overview):
 A token can basically be a word in a natural language. For anonymity, each token is represented by its hash value. 
 Tokens are delimited by the character ‘|’. 
Because of that, since hash of the same word is always the same, I assume that the same ID in different columns
represent the same word.
On 70 mln row file, the script takes approximately 23 minutes to run.
"""

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="Name of the file from which to "
                                               "create o vocabulary file.")
parser.add_argument("-c", "--columns",
                    type=lambda c: c.split(","),
                    default="AdKeyword_tokens,AdTitle_tokens,AdDescription_tokens,Query_tokens",
                    help="Text column from which to create vocabulary. "
                         "Comma separated: for example 'AdTitle_tokens,Query_tokens")
parser.add_argument("-v", "--vocab_file", type=str, default="data/vocab.txt")
parser.add_argument("-wc", "--word_count_file", type=str, default="data/word_count.json")

args = parser.parse_args()
print(args)

word_counter = Counter()

with open(args.filename, newline="") as csv_file:
    reader = csv.reader(csv_file, delimiter="\t")
    columns = next(reader)
    # Figure out position of each chosen column
    column_position = {column: position for position, column in enumerate(columns) if column in args.columns}
    for row in tqdm(reader):
        for column, position in column_position.items():
            tokens = row[position].split("|")
            word_counter.update(tokens)
    # Sort counter by most popular words
    word_count_list = sorted(word_counter.items(), key=lambda item: item[1], reverse=True)
    print("Total number of words", len(word_count_list))
    print("Ten most popular words and their count: ", *word_count_list[:10])
    words = [word + "\n" for word, _ in word_count_list]
    with open(args.vocab_file, "w") as vocab_file:
        vocab_file.writelines(words)
    with open(args.word_count_file, "w") as word_count_file:
        json.dump(OrderedDict(word_count_list), word_count_file)  # json will be ordered by count
