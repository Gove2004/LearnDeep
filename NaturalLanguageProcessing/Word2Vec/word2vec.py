import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from corpus import build_vocab, load_train_corpus


# How many words on each side of the center word count as "context".
WINDOW_SIZE = 2
# Size of the learned word vectors (D). Larger = more capacity, slower to train.
EMBED_DIM = 32
# How many random "wrong" words to draw per training example (the K in negative sampling).
NEGATIVE_SAMPLES = 5
BATCH_SIZE = 128
# One epoch = one full pass over all (context, target) pairs.
EPOCHS = 300
LEARNING_RATE = 0.01
# How many neighbours / analogy answers to print.
TOP_K = 5
# Tiny value added to denominators to avoid dividing by zero.
EPS = 1e-8
SEED = 42


class CBOW(nn.Module):
    """Continuous Bag-of-Words with negative sampling.

    Idea: given the surrounding words (the "context"), predict the word in the middle.
    Two embedding tables are needed because a word plays two different roles:
      - in_embed  : the word AS INPUT  (its vector when it appears in a context)
      - out_embed : the word AS OUTPUT (its vector when it is the word being predicted)
    These are NOT the same vector. After training we keep only in_embed as the
    word representation; out_embed is a training-time helper.
    """

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # Shape of both tables: (vocab_size, embed_dim). Each row is one word's vector.
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        # Standard word2vec init: small uniform values so training starts gently.
        bound = 0.5 / embed_dim
        nn.init.uniform_(self.in_embed.weight, -bound, bound)
        # Output table starts at zero (as in the original word2vec implementation).
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, context, target, negatives):
        # context   : (N, L)  N = batch size, L = number of context words (padded)
        # target    : (N,)    the true center word for each example
        # negatives : (N, K)  K randomly drawn words that did NOT appear as the target

        # Step 1: average the context word vectors -> the "hidden" representation.
        # (N, L, D) -> mean over L -> (N, D)
        h = self.in_embed(context).mean(dim=1)

        # Step 2: positive example. We want the model to say "yes, this pair is real".
        # (N, D) * (N, D) summed over D -> (N, 1) dot product per example.
        pos = self.out_embed(target)
        pos_score = (h * pos).sum(dim=1, keepdim=True)
        # logsigmoid(x) = log(1 / (1 + e^-x)). Maximising it pushes pos_score UP.
        pos_loss = F.logsigmoid(pos_score)

        # Step 3: negative examples. We want the model to say "no, these are fake".
        # bmm = batched matrix multiply: (N, K, D) x (N, D, 1) -> (N, K, 1) -> (N, K)
        neg = self.out_embed(negatives)
        neg_score = torch.bmm(neg, h.unsqueeze(2)).squeeze(2)
        # logsigmoid(-x) is high when neg_score is very negative.
        # Sum over the K negatives, keep dim -> (N, 1).
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1, keepdim=True)

        # We maximise (pos_loss + neg_loss), which equals minimising its negative.
        # Averaging over the batch gives a single scalar loss for backprop.
        return -(pos_loss + neg_loss).mean()


def build_pairs(sentences, word_to_id, window_size=WINDOW_SIZE):
    """Turn raw sentences into (context, target) training pairs.

    For every word we collect its neighbours within `window_size` as the context
    and the word itself as the target. Example with window_size=2:

        sentence : the quick brown fox jumps
        target   : brown
        context  : [the, quick, fox, jumps]

    Returns a list of (context_ids, target_id). The context length varies at the
    edges of a sentence, so it is stored as a plain list and padded later.
    """
    pairs = []
    for sentence in sentences:
        # Drop words that are not in the vocabulary, then work with integer ids.
        ids = [word_to_id[w] for w in sentence if w in word_to_id]
        for i, center in enumerate(ids):
            # Clamp the window so it never runs off either end of the sentence.
            lo = max(0, i - window_size)
            hi = min(len(ids), i + window_size + 1)
            # Everything in the window except the center word itself.
            context = [ids[j] for j in range(lo, hi) if j != i]
            if context:
                pairs.append((context, center))
    return pairs


def negative_sampling_table(freq, word_to_id, table_size=100000, power=0.75):
    """Build a lookup table used to draw random "negative" words.

    Frequent words should be drawn more often than rare ones, but not in direct
    proportion to their frequency -- otherwise words like "the" would dominate
    and the model would learn almost nothing. The original word2vec paper raises
    the frequency to the power 0.75, which softens that skew.

    We pre-compute a big table of word ids sampled with those probabilities, so
    that later we can just pick random slots instead of recomputing probabilities.
    """
    # Sort by id so the i-th entry of `counts` matches word id i.
    counts = np.array([freq[w] for w, _ in sorted(word_to_id.items(), key=lambda kv: kv[1])],
                      dtype=np.float64)
    # f^0.75 : the 0.75 exponent is the standard word2vec smoothing.
    weights = counts ** power
    probs = weights / weights.sum()
    # Sample table_size word ids according to `probs`.
    table = np.random.choice(len(counts), size=table_size, p=probs)
    return torch.from_numpy(table).long()


def make_batches(pairs, table, batch_size=BATCH_SIZE, negative_samples=NEGATIVE_SAMPLES):
    """Yield shuffled mini-batches of (context, target, negatives).

    Because context lengths differ, each batch is padded with zeros up to the
    longest context in that batch. Id 0 is a real word, but since we take the
    mean over all positions it only slightly dilutes the average, and word2vec
    tolerates that in practice.
    """
    order = np.random.permutation(len(pairs))
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        # Pad every context in this chunk to the same width so it fits one tensor.
        max_len = max(len(pairs[i][0]) for i in chunk)
        context = torch.zeros(len(chunk), max_len, dtype=torch.long)
        target = torch.zeros(len(chunk), dtype=torch.long)
        for row, idx in enumerate(chunk):
            ctx, tgt = pairs[idx]
            context[row, :len(ctx)] = torch.tensor(ctx, dtype=torch.long)
            target[row] = tgt
        # For each example draw K random word ids from the sampling table.
        negatives = table[torch.randint(0, len(table), (len(chunk), negative_samples))]
        yield context, target, negatives


def normalize(vectors):
    """Scale every row to unit length so a dot product becomes cosine similarity.

    Cosine similarity ignores vector magnitude and only compares direction, which
    is what we want when measuring how close two words are in meaning.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + EPS)


def most_similar(vectors, id_to_word, word_to_id, word, top_k=TOP_K):
    """Return the top_k words whose vectors point most in the same direction.

    Since rows are unit length, the dot product `unit @ unit[word]` IS the cosine
    similarity for every word at once -- a single matrix-vector product.
    """
    if word not in word_to_id:
        return []
    unit = normalize(vectors)
    # (V, D) @ (D,) -> (V,) : similarity of every word against the query word.
    scores = unit @ unit[word_to_id[word]]
    # argsort ascending, so negate to get descending (most similar first).
    order = np.argsort(-scores)
    out = []
    for idx in order:
        # Skip the query word itself, which is always the closest match.
        if idx == word_to_id[word]:
            continue
        out.append((id_to_word[int(idx)], float(scores[idx])))
        if len(out) >= top_k:
            break
    return out


def analogy(vectors, id_to_word, word_to_id, a, b, c, top_k=3):
    """Solve "a is to b as c is to ?" using vector arithmetic.

    We compute  b - a + c  and look for the nearest word to the result.
    Example: a=man, b=king, c=woman  ->  king - man + woman  ->  queen.
    """
    for w in (a, b, c):
        if w not in word_to_id:
            return []
    unit = normalize(vectors)
    target = unit[word_to_id[b]] - unit[word_to_id[a]] + unit[word_to_id[c]]
    # Re-normalize, otherwise the comparison is biased by the vector's length.
    target = target / (np.linalg.norm(target) + EPS)
    scores = unit @ target
    # Exclude the three input words from the results.
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
    """Distributed Representation by Inference (CBOW + Negative Sampling)"""
    # Fix all random seeds so runs are reproducible.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1) Load the corpus and build the word <-> id mappings.
    sentences = load_train_corpus()
    word_to_id, id_to_word, freq = build_vocab(sentences)
    vocab_size = len(word_to_id)

    # 2) Pre-compute the training pairs and the negative-sampling table once.
    pairs = build_pairs(sentences, word_to_id)
    table = negative_sampling_table(freq, word_to_id)
    print(f"[word2vec] vocab={vocab_size}  pairs={len(pairs)}  embed_dim={EMBED_DIM}")

    model = CBOW(vocab_size, EMBED_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3) Training loop: repeatedly shuffle the pairs and do gradient descent.
    for epoch in range(1, EPOCHS + 1):
        total_loss, batches = 0.0, 0
        for context, target, negatives in make_batches(pairs, table):
            loss = model(context, target, negatives)
            # Standard PyTorch update: clear old grads, backprop, step the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        print(f"[word2vec] epoch {epoch:>4}/{EPOCHS}  loss={total_loss / max(batches, 1):.4f}")

    # 4) The input table is the actual word embedding we keep.
    # (vocab_size, embed_dim) -> row i is the vector for word id i.
    vectors = model.in_embed.weight.detach().numpy()
    print(f"[word2vec] embedding shape={vectors.shape}")

    # 5) Sanity checks: nearest neighbours and a couple of analogies.
    for word in ("dog", "cat", "king", "france"):
        if word not in word_to_id:
            continue
        print(f"[word2vec] most similar to '{word}':")
        for w, score in most_similar(vectors, id_to_word, word_to_id, word):
            print(f"    {w:<12} {score:.4f}")

    print("[word2vec] analogy king - man + woman = ?")
    for w, score in analogy(vectors, id_to_word, word_to_id, "king", "man", "woman"):
        print(f"    {w:<12} {score:.4f}")

    print("[word2vec] analogy paris - france + italy = ?")
    for w, score in analogy(vectors, id_to_word, word_to_id, "paris", "france", "italy"):
        print(f"    {w:<12} {score:.4f}")
