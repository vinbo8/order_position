import torch
import numpy as np
import random
from fairseq.data.data_utils import collate_tokens
from helpers import load_shuffled_model
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from plotly.colors import diverging, sequential

if __name__ == '__main__':
    num_samples = 32
    experiments = ['roberta.base.orig', 'roberta.base.shuffle.n1', 'roberta.base.shuffle.n4', 'roberta.base.shuffle.corpus']
    map_width = 48
    titles = ['word-word', 'word-position', 'position-position']
    fig = make_subplots(rows=len(experiments), cols=3)
    for r, name in enumerate(experiments):
        model = load_shuffled_model(f'./models/{name}')
        dump = [i.split(" ") for i in open("viz/wiki.valid.tokens", "r").read().split("\n")]
        dump = [i for i in dump if len(i) > map_width]
        random.shuffle(dump)
        print(f"total = {len(dump)}")
        dump = [" ".join(i) for i in dump[:num_samples]]

        wi_wj, wi_pj, pi_wj, pi_pj = np.zeros((map_width, map_width)), np.zeros((map_width, map_width)), np.zeros((map_width, map_width)), np.zeros((map_width, map_width))
        tokens = collate_tokens([model.encode(i) for i in dump], pad_idx=1)
        word_embed = model.model.encoder.sentence_encoder.embed_tokens.weight[tokens]
        position_embed = model.model.encoder.sentence_encoder.embed_positions.weight[torch.arange(tokens.size(1)).expand(num_samples, -1)]

        # word_key = model.encoder.layer[0].attention.self.key(word_embed)
        # word_query = model.encoder.layer[0].attention.self.query(word_embed)
        # position_key = model.encoder.layer[0].attention.self.key(position_embed)
        # position_query = model.encoder.layer[0].attention.self.query(position_embed)

        for i in range(map_width):
            for j in range(map_width):
                # wi_wj[i, j] = (word_query[:, i] @ word_key[:, j].T).diagonal().sum()
                # wi_pj[i, j] = (word_query[:, i] @ position_key[:, j].T).diagonal().sum()
                # pi_wj[i, j] = (position_query[:, i] @ word_key[:, j].T).diagonal().sum()
                # pi_pj[i, j] = (position_query[:, i] @ position_key[:, j].T).diagonal().sum()
                wi_wj[i, j] = (word_embed[:, i] @ word_embed[:, j].T).diagonal().sum()
                wi_pj[i, j] = (word_embed[:, i] @ position_embed[:, j].T).diagonal().sum()
                pi_wj[i, j] = (position_embed[:, i] @ word_embed[:, j].T).diagonal().sum()
                pi_pj[i, j] = (position_embed[:, i] @ position_embed[:, j].T).diagonal().sum()

        wi_wj /= num_samples
        wi_pj /= num_samples
        pi_wj /= num_samples
        pi_pj /= num_samples

        for c, i in enumerate([wi_wj, wi_pj, pi_pj]):
            heatmap = go.Heatmap(z=i, zmin=-1, zmax=1, colorscale=diverging.BrBG_r,
                                 colorbar=dict(thickness=30, tickfont_size=24, tickfont_family='Times New Roman'))
            fig.add_trace(heatmap, row=r+1, col=c+1)

    # fig.update_layout(title='', width=800, height=800)
    fig.update_xaxes(title_font_family='Times New Roman', title_font_size=28,
                     tickfont_size=24, tickfont_family='Times New Roman')
    fig.update_yaxes(title_font_family='Times New Roman', title_font_size=28,
                     tickfont_size=24, tickfont_family='Times New Roman')
    fig.update_annotations(font_family="Times New Roman", font_size=28)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()