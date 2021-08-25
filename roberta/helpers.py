import os
import torch
from fairseq.models.roberta import RobertaModel


def load_shuffled_model(path):
    base_model = RobertaModel.from_pretrained(os.path.join(os.path.split(path)[0], 'roberta.base'))
    current_state_dict = torch.load(os.path.join(path, 'model.pt'))['model']
    new_state_dict = {}
    for k in current_state_dict.keys():
        print(k)
        new_state_dict[k.replace('encoder', 'decoder', 1)] = current_state_dict[k]

    base_model.model.load_state_dict(new_state_dict)
    return base_model

