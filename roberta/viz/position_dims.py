import torch
import numpy as np
import tqdm
import random
import transformers
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from plotly.colors import diverging, sequential, qualitative

if __name__ == '__main__':
    TNR = "Times New Roman"
    num_samples = 500
    map_width = 514
    neurons = [0, 4, 8, 16]
    embeds = ['orig', 'shuffle.n1', 'shuffle.n2', 'shuffle.n4', 'shuffle.corpus']
    fig = make_subplots(cols=1, rows=2, subplot_titles=['mean', 'std'], shared_yaxes=True, vertical_spacing=0)

    for c, embed in enumerate(embeds):
        model = torch.load(f'models/roberta.base.{embed}/model.pt')
        # for n, neuron in enumerate(neurons):
        x_labels = torch.arange(0, map_width)
        position_embed = model['model']['encoder.sentence_encoder.embed_positions.weight']
        position_embed = position_embed.detach().numpy()

        palette = qualitative.Safe
        scatter = fig.add_trace(go.Scatter(x=x_labels, y=position_embed.mean(axis=-1),
                             name=embed, marker=dict(color=palette[c]), showlegend=True), row=1, col=1)
        scatter = fig.add_trace(go.Scatter(x=x_labels, y=position_embed.std(axis=-1),
                             name=embed, marker=dict(color=palette[c]), showlegend=False), row=2, col=1)

    fig.update_layout(template='plotly_white',
                      legend=dict(font=dict(size=24, family=TNR)))
    fig.update_xaxes(showticklabels=True, tickvals=[0, 512], tickfont_size=28, tickfont_family=TNR)
    fig.update_yaxes(showticklabels=True, tickmode='array', tickvals=[-0.4, -0.2, 0, 0.2, 0.4],
                     tickfont_size=28, tickfont_family=TNR)
    fig.update_annotations(font=dict(size=28, family=TNR))
    fig.show()
