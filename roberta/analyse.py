from datasets import load_dataset
from collections import Counter
import tokenizers
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numpy.random import binomial
import random
from helpers import load_shuffled_model

# model = load_shuffled_model('./models/roberta.base.orig')
# print(len(model.bpe.bpe.encoder.keys()))
tok = tokenizers.ByteLevelBPETokenizer.from_file('vocab/vocab.json', 'vocab/merges.txt')
counter = 0
d1, d2 = 0, 0
d = []
lengths = []
fig = make_subplots(1, 1)
dataset = load_dataset('bookcorpus', split='train[:10%]')
for line in dataset["text"]:
    if counter == 100000:
        break

    counter += 1
    line = tok.encode(line).ids
    random.shuffle(line)
    d.extend(line)
    lengths.append(len(line))

random.shuffle(d)
d = np.array(d)
odd = d[d % 2 == 0].tolist()
even = d[d % 2 != 0].tolist()
c = Counter(lengths)

counter = 0
for line in dataset["text"]:
    if counter == 100000:
        break

    line = tok.encode(line).ids
    length = len(line)
    counter += 1

    # > 17: length median
    # > 26: token count median
    if length > 26:
        bank = odd; spare = even
    else:
        bank = even; spare = odd

    build = []
    for l in range(length):
        jic = (np.random.randint(0, 5) == 0)
        if jic:
            try:
                build.append(spare.pop())
            except IndexError:
                build.append(bank.pop())
        else:
            try:
                build.append(bank.pop())
            except IndexError:
                build.append(spare.pop())

    print(" ".join(map(str, build)))

    # counter += 1
    # d[d > 4999] = 4999

    # print(" ".join(map(str, d)))

