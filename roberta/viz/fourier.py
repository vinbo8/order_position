import torch
import math
from scipy.fft import fft, fftfreq
import numpy as np
import tqdm
import random
import random
import transformers
import plotly.graph_objects as go
from scipy.stats import shapiro

from plotly.subplots import make_subplots
from plotly.colors import diverging, sequential, qualitative

if __name__ == '__main__':
    TNR = "Times New Roman"
    num_samples = 500
    map_width = 32
    embeds = ['roberta.base.orig', 'roberta.base.shuffle.n1', 'sin']
    dims = random.sample(range(0, 512), 4)
    fig = make_subplots(cols=len(dims), rows=3, row_titles=['Base', '1-gram shuffled', 'sin'], column_titles=list(map(str, dims)), shared_yaxes=True, vertical_spacing=0)

    for r, embed in enumerate(embeds):
        if embed != 'sin':
            model = torch.load(f'models/{embed}/model.pt')
            position_embed = model['model']['encoder.sentence_encoder.embed_positions.weight']
            position_embed = position_embed.detach()
        else:
            half_dim = 768 // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
            emb = torch.arange(514, dtype=torch.float).unsqueeze(
                1
            ) * emb.unsqueeze(0)
            position_embed = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
                514, -1
            )

        palette = qualitative.Safe
        for c, dim in enumerate(dims):
            embed = position_embed[:, dim].numpy()
            ys = np.abs(fft(embed)[:embed.shape[0] // 2])
            xs = np.arange(0, embed.shape[0] // 2)
            fig.add_trace(go.Scatter(x=xs, y=ys), row=r+1, col=c+1)
        # print(shapiro(position_embed.flatten().numpy()))

    fig.update_layout(template='plotly_white',
                      legend=dict(font=dict(size=24, family=TNR)))
    # fig.update_xaxes(showticklabels=True, tickvals=[0, 512], tickfont_size=28, tickfont_family=TNR)
    # fig.update_yaxes(showticklabels=True, tickmode='array', tickvals=[-0.4, -0.2, 0, 0.2, 0.4],
    #                  tickfont_size=28, tickfont_family=TNR)
    fig.update_annotations(font=dict(size=28, family=TNR))
    fig.show()
