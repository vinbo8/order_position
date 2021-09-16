import torch
import numpy as np
import tqdm
import random
import transformers
import plotly.graph_objects as go
from scipy.spatial import procrustes
from scipy.stats import kendalltau, pearsonr, spearmanr
import random

from plotly.subplots import make_subplots
from plotly.colors import diverging, sequential, qualitative
from transformers import BertTokenizer, BertModel

if __name__ == '__main__':
    experiment = 'biling'
    embeds = {'orig': None, 'shuffle.n1': None, 'shuffle.n2': None, 'shuffle.corpus': None, 'noise': None}
    num_embeds = len(embeds.keys())
    fig = make_subplots(rows=1, cols=1)
    corpus = 'wiki'
    map_width = 64
    rows = 32
    for r, embed in enumerate(embeds.keys()):
        if embed == 'noise':
            position_embed = np.random.normal(size=(rows, 514))
        else:
            model = torch.load(f'models/roberta.base.{embed}/model.pt')
            position_embed = model['model']['encoder.sentence_encoder.embed_positions.weight'].detach()
        embeds[embed] = np.zeros((rows, rows))
        for i in [0] + list(range(2, rows)):
            for j in [0] + list(range(2, rows)):
                embeds[embed][i, j] = pearsonr(position_embed[i], position_embed[j])[0]

    rdm = np.zeros((num_embeds, num_embeds))
    for m, i in enumerate(embeds.keys()):
        for n, j in enumerate(embeds.keys()):
            x, y = np.triu_indices(rows, 1)
            flat_i = embeds[i][x, y]
            flat_j = embeds[j][x, y]
            rdm[m, n] = kendalltau(flat_i, flat_j)[0]

    heatmap = go.Heatmap(z=rdm, zmin=0, zmax=1.0, colorscale=sequential.Bluyl,
                         x=list(embeds.keys()), y=list(embeds.keys()))
    fig.add_trace(heatmap, row=1, col=1)
    fig.update_layout(yaxis=dict(autorange='reversed'))
    fig.show()
    print(rdm)