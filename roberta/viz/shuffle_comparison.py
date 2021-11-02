import sys
import torch.nn.functional as F
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


def base_shuffle(some_list):
    randomized_list = some_list[:]
    random.shuffle(randomized_list)
    return randomized_list


def post_bpe_scramble(src_tokens):
    mask = (src_tokens > 2)
    safe_mask = (src_tokens != 1)
    index = torch.stack([F.pad(torch.randperm(i) + 1, (1, mask.size(-1) - i - 1), value=0)
                         for i in torch.count_nonzero(mask, dim=-1)]).type_as(src_tokens)
    src_tokens = torch.gather(src_tokens, 1, index).type_as(src_tokens)
    src_tokens = ((src_tokens - 2) * mask) + 2
    src_tokens = ((src_tokens - 1) * safe_mask) + 1
    src_tokens[:, 0] = 0
    return src_tokens


def main():
    model = load_shuffled_model('./models/roberta.base.orig')
    dataset = load_dataset('bookcorpus', split='train[:1%]')
    fig = make_subplots(rows=1, cols=1)
    base_overlap, deep_overlap, post_overlap = {}, {}, {}
    for line in tqdm.tqdm(dataset["text"][:50000]):
        original = model.encode(' '.join(line.split(' '))).tolist()
        base_shuf = model.encode(' '.join(base_shuffle(line.split(' ')))).tolist()
        deep_shuf = model.encode(' '.join(deep_shuffle(line.split(' ')))).tolist()
        post_shuf = post_bpe_scramble(torch.LongTensor([original])).tolist()[0]
        # original = line.split(' ')
        # base_shuf = base_shuffle(line.split(' '))
        # deep_shuf = deep_shuffle(line.split(' '))

        b_original = set(zip(original, original[1:]))
        b_base_shuf = set(zip(base_shuf, base_shuf[1:]))
        b_deep_shuf = set(zip(deep_shuf, deep_shuf[1:]))
        b_post_shuf = set(zip(post_shuf, post_shuf[1:]))

        l = len(b_original) + 1
        if l not in base_overlap:
            base_overlap[l] = [len(b_base_shuf & b_original) / l]
            deep_overlap[l] = [len(b_deep_shuf & b_original) / l]
            post_overlap[l] = [len(b_post_shuf & b_original) / l]
        else:
            base_overlap[l].append(len(b_base_shuf & b_original) / l)
            deep_overlap[l].append(len(b_deep_shuf & b_original) / l)
            post_overlap[l].append(len(b_post_shuf & b_original) / l)

    base_overlap = list(zip(*sorted({k: np.mean(v) for k, v in base_overlap.items()}.items())))
    base_overlap[0] = base_overlap[0][2:]
    base_overlap[1] = np.array(base_overlap[1])[2:]
    deep_overlap = list(zip(*sorted({k: np.mean(v) for k, v in deep_overlap.items()}.items())))
    deep_overlap[0] = deep_overlap[0][2:]
    deep_overlap[1] = np.array(deep_overlap[1])[2:]
    post_overlap = list(zip(*sorted({k: np.mean(v) for k, v in post_overlap.items()}.items())))
    post_overlap[0] = post_overlap[0][2:]
    post_overlap[1] = np.array(post_overlap[1])[2:]

    fig.add_trace(go.Bar(x=base_overlap[0], y=base_overlap[1], name='random.shuffle'), row=1, col=1)
    fig.add_trace(go.Bar(x=deep_overlap[0], y=deep_overlap[1], name='deep shuffle'), row=1, col=1)
    fig.add_trace(go.Bar(x=post_overlap[0], y=post_overlap[1], name='post shuffle'), row=1, col=1)
    fig.update_layout(barmode='group')
    fig.update_traces(opacity=0.75)
    fig.show()

main()
