import re
from collections import Counter
from pathlib import Path
import urllib.request

PTB_DATASET_PATH = "./Datasets/PTB/"
BASE_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/"
FILES = ["ptb.train.txt", "ptb.valid.txt", "ptb.test.txt"]

def download_ptb_dataset():
    Path(PTB_DATASET_PATH).mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not Path(PTB_DATASET_PATH + f).exists():
            urllib.request.urlretrieve(BASE_URL + f, PTB_DATASET_PATH + f)

def _load_corpus(idx):
    with open(PTB_DATASET_PATH + FILES[idx]) as f:
        text = f.read()

    sentences = [line.split() for line in text.strip().split('\n') if line.strip()]
    return sentences

def load_train_corpus():
    return _load_corpus(0)

def load_valid_corpus():
    return _load_corpus(1)

def load_test_corpus():
    return _load_corpus(2)

def build_vocab(sentences, min_count=5):
    freq = Counter(w for s in sentences for w in s)
    words = [w for w, c in freq.most_common() if c >= min_count]
    word_to_id = {w: i for i, w in enumerate(words)}
    id_to_word = {i: w for w, i in word_to_id.items()}
    return word_to_id, id_to_word, freq

if __name__ == "__main__":
    download_ptb_dataset()
    train_corpus = load_valid_corpus()
    word_to_id, id_to_word, freq = build_vocab(train_corpus, min_count=5)
    print(f"Sentences size: {len(train_corpus)}")
    print(f"Vocabulary size: {len(word_to_id)}")
    print(f"Most common words: {freq.most_common(10)}")
    print(f"Word to ID mapping: {list(word_to_id.items())[:10]}")
    print(f"ID to Word mapping: {list(id_to_word.items())[:10]}")
    print(f"Frequency of '<unk>': {freq['<unk>']}")
    print(f"Frequency of 'N': {freq['N']}")
