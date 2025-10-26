# Bigram Modeling

Predicts the next word using only the previous single word. Approximates the probability of a sentence $(w_1, w_2, \ldots, w_n)$ by:

$$
P(w_1, w_2, \ldots, w_n) \approx P(w_1) \prod_{i=2}^{n} P(w_i \mid w_{i-1})
$$

Instead of conditioning on the entire preceding history, it depends only on the immediately preceding word.

## Probabilities and MLE

Given a training corpus, the Maximum Likelihood Estimation (MLE) for the bigram conditional probability is:

$$
P_{\text{MLE}}(w_j \mid w_i) = \frac{C(w_i, w_j)}{C(w_i)}
$$

where:
- $C(w_i, w_j)$: Count of occurrences of bigram $w_i w_j$ in the corpus.
- $C(w_i)$: Count of occurrences of unigram $w_i$.

The unigram probability is:

$$
P(w) = \frac{C(w)}{N}
$$

where $N$ is the total number of word tokens in the corpus.

## Problems with MLE

Most possible bigrams never occur in a finite corpus. If a bigram has a count of 0, MLE assigns it zero probability, causing any sentence containing it to have zero probability. This is not useful, so **smoothing** is needed to address this issue.

## Smoothing Techniques

Common approaches to smoothing:

### Add-k (Laplace Smoothing)

$$
P_{\text{add-k}}(w_j \mid w_i) = \frac{C(w_i, w_j) + k}{C(w_i) + kV}
$$

- When $k = 1$, this is **Laplace smoothing** (simple, but tends to over-smooth for large vocabularies).
- Smaller $k$ (e.g., $k = 0.01$) often works better.

### Add-ε

Same as add-k, but with a very small value $\epsilon$.

### Good-Turing Smoothing

Adjusts counts based on the "frequency of frequencies." If $N_c$ is the number of n-grams that occur $c$ times, the adjusted count is:

$$
c^* = (c + 1) \frac{N_{c+1}}{N_c}
$$

Then normalize these adjusted counts into probabilities.

### Backoff (Katz Backoff)

Uses discounted MLE probabilities for **seen n-grams** and **backs off** to lower-order models for **unseen ones**, redistributing the leftover probability mass:

$$
P_{\text{backoff}}(w_j \mid w_i) =
\begin{cases}
d_{w_i} \cdot \frac{C(w_i, w_j)}{C(w_i)}, & \text{if } C(w_i, w_j) > 0 \\
\alpha_{w_i} \cdot P(w_j), & \text{otherwise}
\end{cases}
$$

### Interpolation

Blends bigram and unigram probabilities:

$$
P_{\text{interp}}(w_j \mid w_i) = \lambda \cdot P_{\text{MLE}}(w_j \mid w_i) + (1 - \lambda) \cdot P_{\text{MLE}}(w_j)
$$

where $0 \leq \lambda \leq 1$.

### Kneser-Ney Smoothing

One of the most effective modern smoothing methods. It discounts counts and redefines lower-order distributions to favor words that appear in many contexts.

**Absolute discounting version (Bigram case):**

$$
P_{\text{KN}}(w_j \mid w_i) = \frac{\max(C(w_i, w_j) - D, 0)}{C(w_i)} + \lambda(w_i) \cdot P_{\text{continuation}}(w_j)
$$

where:
- $D$: Discount (typically 0.75).
- $\lambda(w_i)$: Backoff weight ensuring probabilities sum to 1.
- $P_{\text{continuation}}(w_j)$: Probability of $w_j$ appearing as a novel continuation:

$$
P_{\text{continuation}}(w_j) = \frac{\text{Number of unique } w_i \text{ such that } C(w_i, w_j) > 0}{\text{Total number of unique bigram types}}
$$

This approach better captures context diversity—for example, "Francisco" is more likely after "San" than "of," even if both have similar counts.

## Unknown Words (OOV) Handling

- Replace rare words (below a frequency threshold) with a special token `<UNK>` during training.
- At test time, unseen words are mapped to `<UNK>`, avoiding zero probabilities for unknown types.

## Evaluation - Perplexity

Perplexity measures how well a model assigns probability to held-out data. For test tokens $(w_1, \ldots, w_T)$:

$$
\text{PP} = 2^{-\frac{1}{T} \sum_{t=1}^{T} \log_2 P(w_t \mid w_{t-1})}
$$

## Training Pipeline

1. Collect a corpus and split it into train/dev/test sets.
2. Normalize tokens (e.g., lowercase, optionally remove punctuation or keep it).
3. Replace rare words in the training set with `<UNK>`; apply the same mapping to dev/test sets.
4. Count unigrams and bigrams over the training set.
5. Compute smoothed probabilities (e.g., add-k, interpolation, or backoff).
6. Evaluate on the dev set (tune $k$ or $\lambda$), then on the test set; compute perplexity.

---

## Glossary (for future me)

| Term / Short Form               | Meaning / Explanation                                                                                            |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **LM**                          | **Language Model** — a statistical or neural model that assigns probabilities to sequences of words.             |
| **Bigram**                      | A sequence of **2 consecutive words**. In bigram LM, probability of next word depends only on the previous word. |
| **Trigram**                     | A sequence of **3 consecutive words**; probability depends on previous 2 words.                                  |
| **Token**                       | A single unit of text — usually a word, punctuation, or special symbol like `<s>`.                               |
| **Corpus**                      | A large body of text used for training or testing a language model.                                              |
| **Sentence Boundary Tokens**    | `<s>` (start of sentence), `</s>` (end of sentence) — used to mark sentence limits.                              |
| **MLE**                         | **Maximum Likelihood Estimation** — method to estimate probabilities from counts in data.                        |
| **C(w)**                        | Count of unigram `w` in corpus.                                                                                  |
| **C(w_i, w_j)**                 | Count of bigram `(w_i w_j)` in corpus.                                                                           |
| **Sparsity**                    | Most possible bigrams never appear in finite corpus — leads to zero probabilities for unseen pairs.              |
| **Smoothing**                   | Adjusting probabilities to handle unseen events and avoid zero probabilities.                                    |
| **Add-k / Laplace Smoothing**   | Simple smoothing: add a small constant `k` to all counts before normalization.                                   |
| **Add-ε**                       | Add a very small `ε` instead of 1, less aggressive than Laplace.                                                 |
| **Good-Turing**                 | Advanced smoothing that adjusts counts based on frequency-of-frequency.                                          |
| **Backoff / Katz**              | Technique: use discounted MLE for seen n-grams, back off to lower-order n-gram for unseen ones.                  |
| **Interpolation**               | Combine probabilities from higher-order and lower-order models (weighted sum) to avoid zero probs.               |
| **OOV / Unknown Words**         | Words not in training vocabulary — mapped to `<UNK>` token.                                                      |
| **Unigram Probability**         | Probability of a single word occurring: `P(w) = C(w)/N`.                                                         |
| **Conditional Probability**     | Probability of a word given previous word: `P(w_j                                                                |
| **Perplexity (PP)**             | Evaluation metric: how well the LM predicts unseen data; lower = better.                                         |
| **V**                           | Vocabulary size = number of unique tokens in the LM.                                                             |
| **Zipf / Rare words threshold** | Words occurring ≤ threshold frequency are replaced by `<UNK>` for OOV handling.                                  |
| **NLTK**                        | **Natural Language Toolkit** — Python library for text processing & corpora.                                     |
| **Brown Corpus**                | Classic balanced English corpus (~1M words) included in NLTK.                                                    |
| **PTB**                         | **Penn Treebank** — standard corpus for LM and parsing research.                                                 |
| **WikiText-2 / WikiText-103**   | Clean Wikipedia-based corpora for language modeling.                                                             |
| **`<s>`, `</s>`**               | Special tokens representing start and end of a sentence, respectively.                                           |
| **zip(s[:-1], s[1:])**          | Python trick: pairs consecutive words to extract bigrams from a sentence list `s`.                               |
