import math
import nltk
import random
from collections import defaultdict, Counter
from nltk.corpus import brown

nltk.download('brown')

def preprocess_sentences(sentences, lowercase=True):
    # the brown corpus alr comes tokenized.
    processed = []
    for sent in sentences:
        if lowercase:
            s = [w.lower() for w in sent]
        else:
            s = list(sent)

        # these are the sentence boundary tokens
        processed.append(['<s>'] + s + ['</s>'])

    return processed

def build_vocab(train_sents, unk_threshold=1):
    # count tokens and replace rare tokens with <UNK>
    unigram_counts = Counter()

    for s in train_sents:
        unigram_counts.update(s)
    
    vocab = set()
    
    for w, c in unigram_counts.items():
        if c > unk_threshold:
            vocab.add(w)
        
    vocab.add('<UNK>')
    return vocab, unigram_counts

def replace_unk(sentences, vocab):
    new_sents = []
    for s in sentences:
        new_sents.append([w if w in vocab else '<UNK>' for w in s])
    
    return new_sents

class BigramModel:
    
    def __init__(self, smoothing_k=1.0):
        self.k = smoothing_k
        self.bigram_counts = Counter()
        self.unigram_counts = Counter()
        self.vocab = set()
        self.v = 0

    def train(self, sentences):
        for s in sentences:
            self.unigram_counts.update(s)
            self.bigram_counts.update(zip(s[:-1], s[1:]))
        
        self.vocab = set(self.unigram_counts.keys())
        self.v = len(self.vocab)

    def prob(self, prev, word):
        # add-k smoothing
        num = self.bigram_counts[(prev, word)] + self.k
        den = self.unigram_counts[prev] + self.k * self.v

        return num / den

    def sent_logprob(self, sent):
        # sent is token list including <s> and </s>
        logp = 0.0

        for a, b in zip(sent[:-1], sent[1:]):
            p = self.prob(a, b)
            logp += math.log2(p)
        
        return logp

    def perplexity(self, sentences):
        total_tokens = 0
        total_logprob = 0.0
        
        for s in sentences:
            total_tokens += (len(s) - 1) # predict tokens after start
            total_logprob += self.sent_logprob(s)

        avg_logprob = total_logprob / total_tokens
        return 2 * (-avg_logprob)

    def sample(self, max_len=20):
        sent = ['<s>']
        for _ in range(max_len):
            prev = sent[-1]

            # build distribution over vocab
            candidates = list(self.vocab)
            probs = [self.prob(prev, w) for w in candidates]

            # normalize (shld alr sum to 1)
            total = sum(probs)
            probs = [p/total for p in probs]
            next_word = random.choices(candidates, weights=probs, k=1)[0]
            sent.append(next_word)

            if next_word == '</s>':
                break
        
        return ' '.join(sent[1:-1]) # remove <s> and </s>

def main():
    sents = list(brown.sents())
    sents = preprocess_sentences(sents, lowercase=True)

    # shuffle and split
    random.shuffle(sents)
    n = len(sents)

    train = sents[:int(0.8*n)]
    dev = sents[int(0.8*n):int(0.9*n)]
    test = sents[int(0.9*n):]

    # build vocab using train; treat words with freq <= 1 as <UNK>
    vocab, train_unigram_counts = build_vocab(train, unk_threshold=1)
    train = replace_unk(train, vocab)
    dev = replace_unk(dev, vocab)
    test = replace_unk(test, vocab)

    # train model with add-k smoothing (k tuneable)
    model = BigramModel(smoothing_k=1.0)
    model.train(train)

    print("Vocab size: ", model.v)
    print("Train tokens: ", sum(model.unigram_counts.values()))

    # evaluate perplexity on dev/test
    pp_dev = model.perplexity(dev)
    pp_test = model.perplexity(test)

    print(f"Perplexity (dev): {pp_dev:.2f}")
    print(f"Perplexity (test): {pp_test:.2f}")

    # sample
    for i in range(5):
        print("Sample: ", model.sample())

    with open("./01_bigram/output.txt", "w") as f:
        f.write(f"Vocab size: {model.v}\n")
        f.write(f"Train tokens: {sum(model.unigram_counts.values())}\n")
        f.write(f"Perplexity (dev): {pp_dev:.2f}\n")
        f.write(f"Perplexity (test): {pp_test:.2f}\n")
        f.write("Samples:\n")
        for i in range(5):
            f.write(model.sample() + "\n")

if __name__ == "__main__":
    main()
