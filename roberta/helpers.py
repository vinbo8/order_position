import os
import torch
from fairseq.models.roberta import RobertaModel


def load_shuffled_model(path):
    if "nopos" in path:
        model = RobertaModel()
    else:
        return RobertaModel.from_pretrained(path)


model = load_shuffled_model('./models/roberta.base.orig')
print("ok")