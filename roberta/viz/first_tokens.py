import sys
import torch
import numpy as np
import tqdm
from scipy.stats import entropy
import random
import math
from roberta.helpers import load_shuffled_model
from datasets import load_dataset
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

model = load_shuffled_model('./roberta/models/roberta.base.orig')


def shuffle_list(some_list):
    randomized_list = some_list[:]
    for _ in range(10000):
        random.shuffle(randomized_list)
        for a, b in zip(some_list, randomized_list):
            if a == b:
                break
        else:
            return randomized_list
    return None


def main():
    fig = make_subplots(1, 1)
    c = Counter()
    orig_first_tokens = []
    shuf_first_tokens = []

    dataset = load_dataset('bookcorpus', split='train[:1%]')
    for line in tqdm.tqdm(dataset["text"][:50000]):
        orig = line
        shuf = shuffle_list(orig.split(' '))
        if not shuf:
            continue
        shuf = ' '.join(shuf)

        orig = model.encode(orig)[1:-1].numpy()
        shuf = model.encode(shuf)[1:-1].numpy()
        orig_first_tokens.append(orig[0].item())
        shuf_first_tokens.append(shuf[0].item())

    orig_counter = Counter(orig_first_tokens)
    shuf_counter = Counter(shuf_first_tokens)
    bins = len(model.bpe.bpe.encoder.keys())

    print(len(set(orig_counter.keys()) - set(shuf_counter.keys())))
    print(f"{len(shuf_counter.keys())}\t{len(orig_counter.keys())}")
    fig.add_trace(go.Histogram(x=list(orig_counter.keys()), y=[i[1] for i in orig_counter.most_common(bins)],
                               nbinsx=bins, histfunc='sum'), row=1, col=1)
    fig.add_trace(go.Histogram(x=list(shuf_counter.keys()), y=[i[1] for i in shuf_counter.most_common(bins)],
                               nbinsx=bins, histfunc='sum'), row=1, col=1)
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.5)
    fig.show()

main()
