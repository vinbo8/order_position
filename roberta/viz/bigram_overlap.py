import sys
from scipy.stats import pointbiserialr, pearsonr
import torch.nn.functional as F
import torch
import numpy as np
import tqdm
from scipy.stats import entropy
import random
import math
from multiset import Multiset
from roberta.helpers import load_shuffled_model
from datasets import load_dataset
import plotly.graph_objects as go
from plotly.colors import diverging, sequential, qualitative
from plotly.subplots import make_subplots
from collections import Counter


tok = load_shuffled_model('models/roberta.base.orig')


def deep_shuffle(some_list):
    randomized_list = some_list[:]
    for _ in range(10000):
        random.shuffle(randomized_list)
        for a, b in zip(some_list, randomized_list):
            if a == b:
                break
        else:
            return randomized_list
    return randomized_list


def overlap_fn(orig, shuf):
    orig = Multiset(zip(orig, orig[1:]))
    shuf = Multiset(zip(shuf, shuf[1:]))
    return len(shuf & orig), len(shuf)


def get_overlaps(corpus, i1, i2, mode):
    i2 = i1 if not i2 else i2
    lengths, overlaps, overlap_percent = [], [], []
    for sentence in corpus:
        s1, s2 = sentence[i1], sentence[i2]
        orig_s1 = tok.encode(s1).tolist()
        orig_s2 = tok.encode(s2).tolist()
        if mode == 'token':
            shuf_s1 = tok.encode(" ".join(deep_shuffle(s1.split()))).tolist()
            shuf_s2 = tok.encode(" ".join(deep_shuffle(s2.split()))).tolist()
        elif mode == 'bpe':
            shuf_s1 = deep_shuffle(tok.encode(s1).tolist())
            shuf_s2 = deep_shuffle(tok.encode(s2).tolist())
        else:
            print("invalid mode")
            return

        o1, l1 = overlap_fn(orig_s1, shuf_s1)
        o2, l2 = overlap_fn(orig_s2, shuf_s2)
        lengths.append(len(orig_s1) + len(orig_s2))
        overlaps.append(o1 + o2)
        overlap_percent.append((o1 + o2) / (l1 + l2))

    return lengths, overlaps, overlap_percent


def main():
    indices = {'RTE': (1, 2), 'PAWS': (1, 2), 'QQP': (3, 4), 'MNLI': (8, 9,), 'QNLI': (1, 2),
               'MRPC': (3, 4), 'SST-2': (0, None), 'CoLA': (3, None)}
    partitions = ['t_shuf_ft_shuf_test', 'b_shuf_ft_shuf_test']
    palette = qualitative.Plotly
    fig = make_subplots(rows=2, cols=2)

    for r, partition in enumerate(partitions):
        for c, model in enumerate(['shuffle.n1']):
            for color, task in enumerate(['QNLI', 'RTE', 'PAWS', 'QQP', 'SST-2', 'CoLA']):
                token_results = [i.rstrip("\n").split("\t") for i in
                                 open(f"./logs/{partition}/{model}.{task}.tsv").readlines()[:-1]]
                sentences = [i.rstrip("\n").split("\t") for i in open(f"./glue/{task}.tsv").readlines()[1:]]
                if not len(sentences) == len(token_results):
                    continue

                bins = np.linspace(0, 25, 25)
                shuffle_mode = 'bpe' if partition.startswith('b_') else 'token'
                s1, s2 = indices[task]
                scores = []

                result_bool = [i[1] == i[2] for i in token_results]
                lengths, overlaps, overlap_percent = get_overlaps(sentences, s1, s2, shuffle_mode)
                print(pointbiserialr(result_bool, lengths))
                print(pointbiserialr(result_bool, overlaps))
                print(pointbiserialr(result_bool, overlap_percent))
                # overlaps = np.digitize(overlaps, bins)
                # overlaps = np.take(bins, overlaps)
                # gathered = list(zip(result_bool, overlaps))

                # for bin in bins:
                #     total = [i for (i, j) in gathered if j == bin]
                #     correct = [i for i in total if i]
                #     if len(total) == 0:
                #         scores.append(None)
                #     else:
                #         scores.append(len(correct) / len(total))

                # fig.add_trace(go.Bar(x=bins, y=scores, marker={'color': palette[color]}, showlegend=not r, name=task),
                #               row=r+1, col=1)
                # fig.add_trace(go.Scatter(x=bins, y=scores, marker=dict(color=palette[color]),
                #                          showlegend=not r, name=task), row=r+1, col=c+1)
    fig.show()


main()
