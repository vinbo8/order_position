import pickle
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from plotly.colors import diverging, sequential, qualitative


def main():
    tasks = ["CoLA", "RTE"]
    models = ["orig", "shuffle.n1"]
    palette = qualitative.Plotly
    fig = make_subplots(rows=2, cols=6, column_titles=tasks, row_titles=models)
    for c, task in enumerate(tasks):
        for r, model in enumerate(models):
            with open(f"lca/{model}.{task}.lca", "rb") as f:
                _lca = pickle.load(f)

            lca = {".".join(k.split(".")[2:-1]):
                       [i.mean() for i in np.array_split(v, 50)] for (k, v) in _lca.items() if k.endswith("weight")}
            lca['embed_tokens'] = (np.array([v for (k, v) in lca.items() if 'embed_tokens' in k])).mean(axis=0)
            lca['embed_positions'] = (np.array([v for (k, v) in lca.items() if 'embed_positions' in k])).mean(axis=0)
            lca['k_proj'] = (np.array([v for (k, v) in lca.items() if 'k_proj' in k])).mean(axis=0)
            lca['q_proj'] = (np.array([v for (k, v) in lca.items() if 'q_proj' in k])).mean(axis=0)
            lca['v_proj'] = (np.array([v for (k, v) in lca.items() if 'v_proj' in k])).mean(axis=0)
            lca['fc1'] = (np.array([v for (k, v) in lca.items() if 'fc1' in k])).mean(axis=0)
            lca['fc2'] = (np.array([v for (k, v) in lca.items() if 'fc2' in k])).mean(axis=0)

            num_entries = len(list(lca.values())[0])
            indices = np.arange(num_entries)

            for col, key in enumerate(["embed_positions", "embed_tokens", "k_proj", "v_proj", "fc1"]):
                fig.add_trace(go.Scatter(x=indices, y=lca[key], name=key, line=dict(color=palette[col]), showlegend=not r),
                              row=r+1, col=c+1)

    fig.update_yaxes(range=(-40, 0))
    fig.show()

main()
