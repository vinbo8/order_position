import torch
import random
from sklearn.decomposition import PCA

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import diverging, sequential, qualitative

if __name__ == '__main__':
    TNR = "Times New Roman"
    num_samples = 500
    map_width = 32
    embeds = ['roberta.base.orig', 'roberta.base.shuffle.n1']
    dims = list(range(200, 210))
    fig = make_subplots(cols=len(dims), rows=2, row_titles=['Base', '1-gram shuffled'], column_titles=list(map(str, dims)), shared_yaxes=True, vertical_spacing=0)

    for r, embed in enumerate(embeds):
        model = torch.load(f'models/{embed}/model.pt')
        position_embed = model['model']['encoder.sentence_encoder.embed_positions.weight']
        position_embed = position_embed.detach()

        palette = qualitative.Safe
        for c, dim in enumerate(dims):
            fig.add_trace(go.Histogram(x=position_embed[dim].numpy(), nbinsx=500), row=r+1, col=c+1)
        # print(shapiro(position_embed.flatten().numpy()))

    fig.update_layout(template='plotly_white',
                      legend=dict(font=dict(size=24, family=TNR)))
    fig.update_xaxes(range=(-1, 1))
    # fig.update_xaxes(showticklabels=True, tickvals=[0, 512], tickfont_size=28, tickfont_family=TNR)
    # fig.update_yaxes(showticklabels=True, tickmode='array', tickvals=[-0.4, -0.2, 0, 0.2, 0.4],
    #                  tickfont_size=28, tickfont_family=TNR)
    fig.update_annotations(font=dict(size=28, family=TNR))
    fig.show()
