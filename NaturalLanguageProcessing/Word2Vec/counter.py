import numpy as np
from scipy.linalg import svd
from corpus import build_vocab, load_train_corpus


WINDOW_SIZE = 2
EMBED_DIM = 16
TOP_K = 5
EPS = 1e-8


def build_cooccurrence(sentences, word_to_id, window_size=WINDOW_SIZE):
    vocab_size = len(word_to_id)
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    for sentence in sentences:
        ids = [word_to_id[w] for w in sentence if w in word_to_id]
        for i, center in enumerate(ids):
            lo = max(0, i - window_size)
            hi = min(len(ids), i + window_size + 1)
            for j in range(lo, hi):
                if i == j:
                    continue
                matrix[center, ids[j]] += 1.0
    return matrix


def ppmi(matrix, eps=EPS):
    total = matrix.sum()
    row_sum = matrix.sum(axis=1, keepdims=True)
    col_sum = matrix.sum(axis=0, keepdims=True)
    pmi = np.log((matrix * total) / (row_sum * col_sum + eps) + eps)
    return np.maximum(pmi, 0.0)


def reduce_dimension(ppmi_matrix, dim=EMBED_DIM):
    u, s, _ = svd(ppmi_matrix, full_matrices=False)
    dim = min(dim, len(s))
    return u[:, :dim] * s[:dim]


def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + EPS)


def most_similar(vectors, id_to_word, word_to_id, word, top_k=TOP_K):
    if word not in word_to_id:
        return []
    unit = normalize(vectors)
    target = unit[word_to_id[word]]
    scores = unit @ target
    order = np.argsort(-scores)
    out = []
    for idx in order:
        if idx == word_to_id[word]:
            continue
        out.append((id_to_word[int(idx)], float(scores[idx])))
        if len(out) >= top_k:
            break
    return out


def analogy(vectors, id_to_word, word_to_id, a, b, c, top_k=3):
    for w in (a, b, c):
        if w not in word_to_id:
            return []
    unit = normalize(vectors)
    target = unit[word_to_id[b]] - unit[word_to_id[a]] + unit[word_to_id[c]]
    target = target / (np.linalg.norm(target) + EPS)
    scores = unit @ target
    banned = {word_to_id[a], word_to_id[b], word_to_id[c]}
    order = np.argsort(-scores)
    out = []
    for idx in order:
        if int(idx) in banned:
            continue
        out.append((id_to_word[int(idx)], float(scores[idx])))
        if len(out) >= top_k:
            break
    return out


def run_learn(*args, **kwargs):
    """Distributed Representation by Counting"""
    sentences = load_train_corpus()
    word_to_id, id_to_word, freq = build_vocab(sentences)

    print(f"[counter] sentences={len(sentences)}  vocab={len(word_to_id)}")

    cooc = build_cooccurrence(sentences, word_to_id)
    print(f"[counter] co-occurrence matrix shape={cooc.shape}  nonzero={int((cooc > 0).sum())}")

    ppmi_matrix = ppmi(cooc)
    print(f"[counter] PPMI matrix shape={ppmi_matrix.shape}  max={ppmi_matrix.max():.4f}")

    vectors = reduce_dimension(ppmi_matrix, EMBED_DIM)
    print(f"[counter] embedding shape={vectors.shape}")

    for word in ("dog", "cat", "king", "france"):
        if word not in word_to_id:
            continue
        print(f"[counter] most similar to '{word}':")
        for w, score in most_similar(vectors, id_to_word, word_to_id, word):
            print(f"    {w:<12} {score:.4f}")

    print("[counter] analogy king - man + woman = ?")
    for w, score in analogy(vectors, id_to_word, word_to_id, "king", "man", "woman"):
        print(f"    {w:<12} {score:.4f}")

    print("[counter] analogy paris - france + italy = ?")
    for w, score in analogy(vectors, id_to_word, word_to_id, "paris", "france", "italy"):
        print(f"    {w:<12} {score:.4f}")
