import random
import sys
import tqdm
import plotly.graph_objects as go
import torch
from argparse import ArgumentParser
from scipy.stats import pointbiserialr, pearsonr
from roberta.helpers import load_shuffled_model
from plotly.colors import diverging, sequential, qualitative
from plotly.subplots import make_subplots
from fairseq.models.roberta import RobertaModel


def deep_shuffle(some_list):
    if isinstance(some_list, list):
        randomized_list = some_list[:]
    else:
        randomized_list = some_list.split()
    for _ in range(10000):
        random.shuffle(randomized_list)
        for a, b in zip(some_list, randomized_list):
            if a == b:
                break
        else:
            return randomized_list if isinstance(some_list, list) else " ".join(randomized_list)
    return randomized_list if isinstance(some_list, list) else " ".join(randomized_list)


def get_saliencies(roberta, pair, logit):
    roberta.zero_grad()
    roberta.predict('sentence_classification_head', pair)[0][logit].backward()
    positions = roberta.model.encoder.sentence_encoder.embed_positions.weight
    saliencies = positions.grad.T.norm(dim=0).unsqueeze(0).numpy()
    # saliencies = torch.einsum("ij,jk->i", positions.grad, positions.detach().T).unsqueeze(0).numpy()
    return saliencies


def main():
    indices = {'RTE': (1, 2, 3), 'PAWS': (1, 2, 3), 'QQP': (3, 4, 5), 'MNLI': (8, 9, 15), 'QNLI': (1, 2, 3),
               'MRPC': (3, 4, 0), 'SST-2': (0, None, 1), 'CoLA': (3, None, 1)}
    parser = ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--results', type=str)
    parser.add_argument('--glue_dir', type=str)
    parser.add_argument('--task', type=str)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    roberta = RobertaModel.from_pretrained(f'{args.model}', checkpoint_file='checkpoint_best.pt',
                                           data_name_or_path=f'{args.glue_dir}/{args.task}-bin').to(device)
    results = [i.rstrip("\n").split("\t") for i in open(f"{args.results}").readlines()[:-1]]
    sentences = [i.rstrip("\n").split("\t") for i in open(f"{args.glue_dir}/{args.task}.tsv").readlines()[1:]]
    i1, i2, il = indices[args.task]
    correct, saliency = [], []
    for r, (result, sentence) in enumerate(zip(results, sentences)):
        s1, s2, label = sentence[i1], sentence[i2], sentence[il]
        n = roberta.task.label_dictionary.encode_line(label)[0].item() - roberta.task.label_dictionary.nspecial
        orig_pair = roberta.encode(s1, s2)
        orig_saliencies = get_saliencies(roberta, orig_pair, n)
        saliency.append(orig_saliencies.mean())
        correct.append(result[-2] == result[-1])

    print(f"{pearsonr(correct, saliency)}")

main()
