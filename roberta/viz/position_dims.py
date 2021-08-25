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
    map_width = 32
    neurons = [0, 4, 8, 16]
    embeds = ['roberta.base.orig', 'roberta.base.shuffle.n1']
    fig = make_subplots(cols=2, rows=1, subplot_titles=['Base', '1-gram shuffled'], shared_yaxes=True, vertical_spacing=0)

    for c, embed in enumerate(embeds):
        model = torch.load(f'models/{embed}/model.pt')
        for n, neuron in enumerate(neurons):
            x_labels = torch.arange(0, map_width)
            position_embed = model['model']['encoder.sentence_encoder.embed_positions.weight']
            position_embed = position_embed.detach().numpy()

            palette = qualitative.Safe
            scatter = go.Scatter(x=x_labels, y=position_embed[:map_width, neuron],
                                 name=neuron, marker=dict(color=palette[n]), showlegend=not c)
            fig.add_trace(scatter, row=1, col=c+1)

    fig.update_layout(template='plotly_white',
                      legend=dict(font=dict(size=24, family=TNR)))
    fig.update_xaxes(showticklabels=True, tickvals=[0, 512], tickfont_size=28, tickfont_family=TNR)
    fig.update_yaxes(showticklabels=True, tickmode='array', tickvals=[-0.4, -0.2, 0, 0.2, 0.4],
                     tickfont_size=28, tickfont_family=TNR)
    fig.update_annotations(font=dict(size=28, family=TNR))
    fig.show()
