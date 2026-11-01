import numpy as np
from gensim.models import Word2Vec

from corpus import build_vocab, load_train_corpus


EMBED_DIM = 32
WINDOW_SIZE = 2
MIN_COUNT = 1
EPOCHS = 300
NEGATIVE_SAMPLES = 5
TOP_K = 5
EPS = 1e-8
SEED = 42


def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + EPS)


def most_similar(model, word, top_k=TOP_K):
    if word not in model.wv:
        return []
    return [(w, float(s)) for w, s in model.wv.most_similar(word, topn=top_k)]


def analogy(model, a, b, c, top_k=3):
    for w in (a, b, c):
        if w not in model.wv:
            return []
    return [(w, float(s)) for w, s in model.wv.most_similar(positive=[b, c], negative=[a], topn=top_k)]


def run_learn(*args, **kwargs):
    """Distributed Representation with gensim"""
    sentences = load_train_corpus()
    word_to_id, id_to_word, freq = build_vocab(sentences)

    model = Word2Vec(
        sentences=sentences,
        vector_size=EMBED_DIM,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        sg=0,
        negative=NEGATIVE_SAMPLES,
        epochs=EPOCHS,
        seed=SEED,
        workers=1,
    )

    print(f"[mygensim] vocab={len(model.wv)}  vector_size={model.wv.vector_size}")

    for word in ("dog", "cat", "king", "france"):
        if word not in model.wv:
            continue
        print(f"[mygensim] most similar to '{word}':")
        for w, score in most_similar(model, word):
            print(f"    {w:<12} {score:.4f}")

    print("[mygensim] analogy king - man + woman = ?")
    for w, score in analogy(model, "king", "man", "woman"):
        print(f"    {w:<12} {score:.4f}")

    print("[mygensim] analogy paris - france + italy = ?")
    for w, score in analogy(model, "paris", "france", "italy"):
        print(f"    {w:<12} {score:.4f}")
