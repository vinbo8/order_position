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
    whitespace_orig_type = 0
    whitespace_shuf_type = 0
    shuf_minus_orig_tok = []
    shuf_minus_orig_len = []
    with open(sys.argv[1], 'r') as f:
        all_orig_tokens = []
        all_shuf_tokens = []

        dataset = load_dataset('bookcorpus', split='train[:1%]')
        for line in tqdm.tqdm(dataset["text"][:50000]):
            orig = line
            shuf = shuffle_list(orig.split(' '))
            if not shuf:
                continue
            shuf = ' '.join(shuf)

            orig = model.encode(orig)[2:-1].numpy()
            shuf = model.encode(shuf)[2:-1].numpy()

            shuf_minus_orig_tok.append(len(set(shuf) - set(orig)))
            # shuf_minus_orig_len.append(len(shuf) - len(orig))

            all_orig_tokens.extend(list(orig))
            all_shuf_tokens.extend(list(shuf))

    # orig_counter = Counter(all_orig_tokens)
    # shuf_counter = Counter(all_shuf_tokens)
    print(f"{shuf_minus_orig_tok}")
    # shuf_minus_orig_len = np.mean(shuf_minus_orig_len)
    shuf_minus_orig_tok = np.mean(shuf_minus_orig_tok)

    print(f"{shuf_minus_orig_tok}")

    # bins = 1000

    # for (i1, j1), (i2, j2) in zip(orig_counter.most_common(bins), shuf_counter.most_common(bins)):
    #     if ' ' in model.decode(torch.LongTensor([i1])):
    #         whitespace_orig_type += 1
    #     if ' ' in model.decode(torch.LongTensor([i2])):
    #         whitespace_shuf_type += 1
    #
    # print(f"{whitespace_orig_type / len(orig_counter)}\t{whitespace_shuf_type / len(shuf_counter)}")

    # sys.stdout.write(f"{entropy(all_orig_tokens)}\t{entropy(all_shuf_tokens)}")
    # fig.add_trace(go.Histogram(x=np.arange(len(orig_counter)), y=[i[1] for i in orig_counter.most_common(bins)],
    #                            nbinsx=bins, histfunc='sum'), row=1, col=1)
    # fig.add_trace(go.Histogram(x=np.arange(len(shuf_counter)), y=[i[1] for i in shuf_counter.most_common(bins)],
    #                            nbinsx=bins, histfunc='sum'), row=1, col=1)
    # fig.update_layout(barmode='overlay')
    # fig.update_traces(opacity=0.5)
    # fig.show()

main()
