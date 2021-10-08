import torch
import math
import sys
import numpy as np
import transformers
from helpers import load_shuffled_model
import random


def get_invariance(A, width, init):
    N, D = 0, 0
    for l in range(-width, width):
        indices = [(i, j) for i in range(init, width) for j in range(init, width) if j - i == l]
        if l == 0 or not indices:
            continue

        values = [A[i].item() for i in indices]
        N += np.var(values) * len(values)
        D += len(values)

    aiv = N / D
    aiv /= torch.var(A)
    return aiv.item()


def get_monotonicity(A, width, limit):
    N, D = 0, 0
    S = []
    for i in range(0, width):
        S.append([A[i, j].item() for j in range(i, width)])
        S.append([A[i, j].item() for j in range(i, 0, -1)])

    for s in S[:limit]:
        if len(s) <= 1:
            continue
        opr = 0
        for m in range(len(s)):
            for n in range(len(s)):
                if m == n:
                    continue
                opr += 1 if (s[m] - s[n]) * (m - n) > 0 else 0
        opr /= (len(s) ** 2) - len(s)
        N += opr * len(s)
        D += len(s)

    return N / D


def get_symmetry(A, width):
    N, D = 0, 0
    for i in range(width):
        for j in range(i):
            N += abs((A[i, j] - A[j, i]).item())
    D = width * (width - 1) / 2

    return N / D


def main():
    names = ['roberta.base', 'roberta.base.orig', 'roberta.base.shuffle.n1', 'roberta.base.shuffle.corpus', 'bert-base-uncased']
    width = 128
    attempts = 300
    sys.stdout.write(f"model\tinv. w cls\tinv. w/o cls\tmono\tmono, first 20 offsets\n")
    for n in names:
        A = torch.zeros(width, width)
        if 'bert-' in n:
            model = transformers.BertModel.from_pretrained(n)
            words = model.embeddings.word_embeddings
            positions = model.embeddings.position_embeddings.weight.detach()
            q_proj = model.encoder.layer[0].attention.self.query
            k_proj = model.encoder.layer[0].attention.self.key
            token_bank = list(range(model.embeddings.word_embeddings.weight.size(0)))
        else:
            model = load_shuffled_model(f"models/{n}")
            words = model.model.encoder.sentence_encoder.embed_tokens
            positions = model.model.encoder.sentence_encoder.embed_positions.weight.detach()
            q_proj = model.model.encoder.sentence_encoder.layers[0].self_attn.q_proj
            k_proj = model.model.encoder.sentence_encoder.layers[0].self_attn.k_proj
            token_bank = list(model.bpe.bpe.decoder.keys())

        for token in random.sample(token_bank, attempts):
            sentence = torch.LongTensor([token] * width)
            w = words(sentence)
            p = positions[:sentence.size(0)]
            A += q_proj(w + p) @ k_proj(w + p).T

        A /= attempts

        inv_w_cls = get_invariance(A, width, 0)
        inv_wo_cls = get_invariance(A, width, 1)
        mono = get_monotonicity(A, width, 999)
        mono_ltd = get_monotonicity(A, width, 20)
        # sym = get_symmetry(A, width)
        sys.stdout.write(f"{n}\t{inv_w_cls}\t{inv_wo_cls}\t{mono}\t{mono_ltd}\n")


if __name__ == '__main__':
    main()
