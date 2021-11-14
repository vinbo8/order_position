import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import random
import transformers
from fairseq.models.roberta.model import roberta_small_architecture, RobertaModel
from helpers import load_shuffled_model
import plotly.io as pio
from collections import Counter
from datasets import load_dataset
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from plotly.colors import diverging, sequential, qualitative, n_colors, unlabel_rgb, label_rgb

if __name__ == '__main__':
    pio.templates.default = 'plotly_white'
    lo, hi = map(unlabel_rgb, (sequential.Blues[0], sequential.Blues[-1]))
    experiment = 'biling'
    # embeds = ['baseline', 'bpe_scramble', 'token_scramble', "bpe_sinusoidal"]
    # embeds = ['nopos']
    # titles = embeds
    # embeds = ["baseline"]
    embeds = ['orig', 'nopos', 'shuffle.n1', 'shuffle.freeze']
    titles = ['ORIG', 'NOPOS', 'SHUF.N1', 'SHUF.N1 (-POS)']
    layers = [0, 1, 6, 7, 10, 11]
    palette = sequential.RdPu[2:]
    num_embeds = len(embeds)
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles, x_title="Offset", y_title="Frequency (%)", vertical_spacing=0.1,
                        shared_yaxes=True, shared_xaxes=True)
    nheads = 12
    map_width = 32
    dataset = load_dataset('bookcorpus', split='train[:1%]')
    for c, embed in enumerate(embeds):
        heads = {j: [] for j in layers}
        if embed in ['baseline', 'bpe_scramble', 'token_scramble', 'bpe_sinusoidal']:
            model = load_shuffled_model(f'models/roberta.base.orig')
            roberta_small_architecture(model.model.args)
            args, task = model.model.args, model.task
            roberta = RobertaModel.build_model(args, task)
            params = torch.load(f'models/{embed}.42/checkpoint_last.pt')['model']
            roberta.load_state_dict(params, strict=False)
            model.model = roberta

        elif embed == 'shuffle.freeze':
            model = load_shuffled_model(f'models/roberta.base.shuffle.n1')
            torch.nn.init.zeros_(model.model.encoder.sentence_encoder.embed_positions.weight)

        else:
            model = load_shuffled_model(f'models/roberta.base.{embed}')

        model.eval()
        # model = torch.load(f'models/roberta.base.{embed}/model.pt')
        for s in tqdm.tqdm(dataset["text"][:10]):
            stack = model.extract_features(model.encode(s), return_all_hiddens=True)
            for r, layer in enumerate(layers):
                embeds = stack[layer][0]
                n = embeds.size(0)
                q_proj = model.model.encoder.sentence_encoder.layers[layer].self_attn.q_proj
                k_proj = model.model.encoder.sentence_encoder.layers[layer].self_attn.k_proj
                q = q_proj(embeds).view(n, nheads, 64).transpose(0, 1)
                k = k_proj(embeds).view(-1, nheads, 64).transpose(0, 1)
                attn_weights = F.softmax(q @ k.transpose(1, 2), dim=-1)
                distances = (attn_weights.argmax(dim=-1) - torch.arange(0, n).repeat([nheads, 1]))
                for i in range(nheads):
                    heads[layer].extend([abs(n) for n in distances[i].detach().tolist()])

        for r, layer in enumerate(layers):
            counter = sorted(Counter(heads[layer]).items())
            x, y = zip(*counter)
            norm = sum(y)
            y = [(i * 100) / norm for i in y]
            fig.add_trace(go.Scatter(x=x[:10], y=y[:10], mode='lines', showlegend=False,
                                     line=dict(color=palette[r], shape='spline', smoothing=1.2, width=2)),
                          row=(c // 2) + 1, col=(c % 2) + 1)

    tfont = dict(family="Arial", size=20)
    mfont = dict(family="Times New Roman", size=30)
    fig.update_annotations(font_family="Times New Roman", font_size=30)
    fig.update_layout(font=dict(family="Times New Roman", size=42), width=1000, height=1000)
    fig.update_xaxes(title_font=mfont, tickfont=tfont, tickmode='array', tickvals=list(range(0, 9)))
    fig.update_yaxes(title_font=mfont, tickfont=tfont, range=[0, 35], tickmode='array', tickvals=[10, 20, 30])

    fig.show()
