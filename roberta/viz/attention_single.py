import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import random
import transformers
from fairseq.models.roberta.model import roberta_small_architecture, RobertaModel
from helpers import load_shuffled_model
from collections import Counter
from datasets import load_dataset
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from plotly.colors import diverging, sequential, qualitative

if __name__ == '__main__':
    palette = sequential.YlGnBu
    experiment = 'biling'
    embeds = ['bpe_scramble', 'baseline', 'token_scramble']
    layers = [0, 1,  6, 7,  10, 11]
    num_embeds = len(embeds)
    fig = make_subplots(rows=1, cols=len(embeds), column_titles=list(embeds))
    dataset = load_dataset('bookcorpus', split='train[:1%]')

    for c, embed in enumerate(embeds):
        nheads = 12
        map_width = 32
        nsents = 100
        heads = {j: np.array([0] * map_width) for j in layers}
        counts = {j: np.array([0] * map_width) for j in layers}

        if embed in ['baseline', 'bpe_scramble', 'token_scramble']:
            model = load_shuffled_model(f'models/roberta.base.orig')
            roberta_small_architecture(model.model.args)
            args, task = model.model.args, model.task
            roberta = RobertaModel.build_model(args, task)
            params = torch.load(f'models/{embed}.42/checkpoint_last.pt')['model']
            roberta.load_state_dict(params)
            model.model = roberta
            nheads = 1
            nsents = 100

        elif embed == 'shuffle.freeze':
            model = load_shuffled_model(f'models/roberta.base.shuffle.n1')
            torch.nn.init.zeros_(model.model.encoder.sentence_encoder.embed_positions.weight)
        else:
            model = load_shuffled_model(f'models/roberta.base.{embed}')
        model.eval()

        # model = torch.load(f'models/roberta.base.{embed}/model.pt')
        for s in tqdm.tqdm(dataset["text"][:nsents]):
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
                for m in range(attn_weights.size(-1)):
                    for n in range(attn_weights.size(-1)):
                        if abs(m - n) < map_width:
                            heads[layer][abs(m - n)] += attn_weights[:, m, n].sum().item()
                            counts[layer][abs(m - n)] += nheads

        for r, layer in enumerate(layers):
            fig.add_trace(go.Scatter(x=list(range(map_width)), y=heads[layer] / counts[layer], mode='lines',
                                     line=dict(color=palette[r], shape='spline', smoothing=1.3)),
                          row=1, col=c+1)

    fig.update_layout(xaxis_title="Head", yaxis_title="Mean distance")
    fig.update_yaxes(range=[0, 0.5])

    fig.show()
