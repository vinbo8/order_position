import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import random
import transformers
from helpers import load_shuffled_model
from datasets import load_dataset
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from plotly.colors import diverging, sequential, qualitative

if __name__ == '__main__':
    palette = qualitative.T10
    experiment = 'biling'
    embeds = ['orig', 'shuffle.n1', 'nopos']
    layers = [0, 7, 11]
    num_embeds = len(embeds)
    fig = make_subplots(rows=1, cols=len(embeds), column_titles=list(embeds))
    map_width = 64
    dataset = load_dataset('bookcorpus', split='train[:1%]')
    for c, embed in enumerate(embeds):
        heads = {i: {j: [] for j in range(len(layers))} for i in range(12)}
        model = load_shuffled_model(f'models/roberta.base.{embed}')
        model.eval()
        # model = torch.load(f'models/roberta.base.{embed}/model.pt')
        for s in tqdm.tqdm(dataset["text"][:100]):
            stack = model.extract_features(model.encode(s), return_all_hiddens=True)
            for r, layer in enumerate(layers):
                embeds = stack[layer][0]
                n = embeds.size(0)
                q_proj = model.model.encoder.sentence_encoder.layers[layer].self_attn.q_proj
                k_proj = model.model.encoder.sentence_encoder.layers[layer].self_attn.k_proj
                q = q_proj(embeds).view(n, 12, 64).transpose(0, 1)
                k = k_proj(embeds).view(-1, 12, 64).transpose(0, 1)
                attn_weights = F.softmax(q @ k.transpose(1, 2), dim=-1)
                distances = (attn_weights.argmax(dim=-1) - torch.arange(0, n).repeat([12, 1]))
                for i in range(12):
                    heads[i][r].extend([abs(n) for n in distances[i].detach().tolist()])

        for r, layer in enumerate(layers):
            fig.add_trace(go.Bar(x=list(heads.keys()), y=[np.mean(i[r]) for _, i in heads.items()], name=f"layer {r}",
                                 marker_color=palette[r], showlegend=not c), row=1, col=c+1)

    fig.update_layout(xaxis_title="Head", yaxis_title="Mean distance")

    fig.show()
