import torch
import numpy as np
import tqdm
import random
import transformers
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from plotly.colors import diverging, sequential
from transformers import BertTokenizer, BertModel

if __name__ == '__main__':
    map_width = 49
    embeds = ['roberta.base.orig', 'roberta.base.shuffle.n1']
    fig = make_subplots(rows=1, cols=2, subplot_titles=embeds, shared_xaxes=True, shared_yaxes=True)

    for c, embed in enumerate(embeds):
        model = torch.load(f'models/{embed}/model.pt')
        position_embed = model['model']['encoder.sentence_encoder.embed_positions.weight']

        pos_correl = torch.zeros(map_width, map_width)
        for i in range(map_width):
            for j in range(map_width):
                pos_correl[i, j] = (position_embed[i, :] @ position_embed[j, :].T)

        fig.add_trace(go.Heatmap(z=pos_correl, zmin=0, zmax=2, colorscale=diverging.curl_r), row=1, col=c+1)

    fig.update_xaxes(showticklabels=False, tickfont_size=24, tickvals=[0, 48], tickfont_family='Times New Roman')
    fig.update_yaxes(showticklabels=False, tickfont_size=24, tickvals=[0, 48], tickfont_family='Times New Roman')
    # fig.update_annotations(font_family='Courier New', font_size=28)
    fig.show()