import os
from torch import nn
import json
import torch
import pickle
from fairseq.models.roberta import RobertaModel

use_cuda = torch.cuda.is_available()


def load_and_invert(path):
    base_model = RobertaModel.from_pretrained(os.path.join(os.path.split(path)[0], "roberta.base.orig"))
    args, task = base_model.model.args, base_model.task
    new_model = RobertaModel.build_model(args, task)
    state_dict = torch.load(os.path.join(path, "model.pt"))
    state_dict["model"]["encoder.sentence_encoder.embed_positions.weight"] = \
        base_model.model.encoder.sentence_encoder.embed_positions.weight
    embed_dim = base_model.model.encoder.sentence_encoder.embed_positions.weight.size(-1)
    mean, std = 0, embed_dim ** -0.5
    nn.init.normal_(state_dict["model"]["encoder.sentence_encoder.embed_positions.weight"], mean=mean, std=std)
    new_model.load_state_dict(state_dict['model'])
    base_model.model = new_model
    if use_cuda:
        base_model.cuda()
    return base_model


def load_shuffled_model(path):
    if "nopos" in path:
        base_model = RobertaModel.from_pretrained(os.path.join(os.path.split(path)[0], "roberta.base.orig"))
        args, task = base_model.model.args, base_model.task
        args.no_token_positional_embeddings = True
        new_model = RobertaModel.build_model(args, task)
        state_dict = torch.load(os.path.join(path, "model.pt"))
        new_model.load_state_dict(state_dict['model'])
        base_model.model = new_model
        if use_cuda:
            base_model.cuda()
        return base_model

    else:
        model = RobertaModel.from_pretrained(path)
        if use_cuda:
            model.cuda()
        return model
