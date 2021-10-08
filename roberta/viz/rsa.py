import torch
import numpy as np
import tqdm
import random
import transformers
from helpers import load_shuffled_model
from datasets import load_dataset
import plotly.graph_objects as go
from scipy.spatial import procrustes
from scipy.stats import kendalltau, pearsonr, spearmanr
import random

from plotly.subplots import make_subplots
from plotly.colors import diverging, sequential, qualitative
from transformers import BertTokenizer, BertModel

if __name__ == '__main__':
    experiment = 'biling'
    embeds = {'orig': None, 'shuffle.n1': None, 'shuffle.n2': None, 'shuffle.corpus': None}
    num_embeds = len(embeds.keys())
    fig = make_subplots(cols=3, rows=len(embeds.keys()), row_titles=list(embeds.keys()))
    map_width = 64
    dataset = load_dataset('bookcorpus', split='train[:1%]')
    for r, embed in enumerate(embeds.keys()):
        model = load_shuffled_model(f'models/roberta.base.{embed}')
        # model = torch.load(f'models/roberta.base.{embed}/model.pt')
        position_embed = model.model.encoder.sentence_encoder.embed_positions.weight.detach()
        q_proj = model.model.encoder.sentence_encoder.layers[0].self_attn.q_proj
        k_proj = model.model.encoder.sentence_encoder.layers[0].self_attn.k_proj
        w2w, w2p, p2w, p2p = np.zeros((map_width, map_width)), np.zeros((map_width, map_width)), \
                             np.zeros((map_width, map_width)), np.zeros((map_width, map_width))
        mask = np.zeros((map_width, map_width))
        total = 0
        for s in tqdm.tqdm(dataset["text"][:10000]):
            s = model.encode(s)
            l = min(map_width, len(s))
            w = model.model.encoder.sentence_encoder.embed_tokens(s)[:l]
            p = model.model.encoder.sentence_encoder.embed_positions.weight[:l]
            pad_len = map_width - w.size(0)
            w2p += np.pad((k_proj(w) @ q_proj(p).T).detach().numpy(), ((0, pad_len), (0, pad_len)))
            p2w += np.pad((q_proj(w) @ k_proj(p).T).detach().numpy(), ((0, pad_len), (0, pad_len)))
            w2w += np.pad((q_proj(w) @ k_proj(w).T).detach().numpy(), ((0, pad_len), (0, pad_len)))
            mask += np.pad(np.ones((w.size(0), w.size(0))), ((0, pad_len), (0, pad_len)))

        # print(total)
        w2p /= mask
        p2w /= mask
        w2w /= mask
        for i in range(map_width):
            for j in range(map_width):
                p2p[i, j] = position_embed[i] @ position_embed[j]

        fig.add_trace(go.Heatmap(z=w2w, x=list(range(map_width)), y=list(range(map_width)), zmin=-1, zmax=0,
                      colorscale=sequential.Blues.reverse()), row=r+1, col=1)
        fig.add_trace(go.Heatmap(z=w2p, x=list(range(map_width)), y=list(range(map_width)), zmin=-1, zmax=0,
                      colorscale=sequential.Blues.reverse()), row=r+1, col=2)
        fig.add_trace(go.Heatmap(z=p2w, x=list(range(map_width)), y=list(range(map_width)), zmin=-3, zmax=-1,
                      colorscale=sequential.Blues.reverse()), row=r+1, col=3)

    # fig.update_layout(yaxis=dict(autorange='reversed'))
    # rdm = np.zeros((num_embeds, num_embeds))
    # for m, i in enumerate(embeds.keys()):
    #     for n, j in enumerate(embeds.keys()):
    #         x, y = np.triu_indices(rows, 1)
    #         flat_i = embeds[i][x, y]
    #         flat_j = embeds[j][x, y]
    #         rdm[m, n] = kendalltau(flat_i, flat_j)[0]
    #
    # heatmap = go.Heatmap(z=rdm, zmin=0, zmax=1.0, colorscale=sequential.Bluyl,
    #                      x=list(embeds.keys()), y=list(embeds.keys()))
    # fig.add_trace(heatmap, row=1, col=1)
    fig.show()
