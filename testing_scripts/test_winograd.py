from roberta.helpers import load_shuffled_model
from fairseq.models.roberta import RobertaModel
from examples.roberta.wsc import wsc_utils  # also loads WSC task and criterion
import torch
import argparse

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')


def main():
    parser = argparse.ArgumentParser(description="generate token embeddings from corpus");
    parser.add_argument('-d', "--dataset_path", type=str);
    parser.add_argument('-m', "--model_path", type=str);
    parser.add_argument('-s', "--split", type=str);
    arguments = parser.parse_args();

    # Load pre-trained model (weights) and test o
    #model = load_shuffled_model('roberta/models/roberta.base.orig')
    roberta = RobertaModel.from_pretrained(arguments.model_path, 'model.pt', arguments.dataset_path)
    roberta.cuda()
    nsamples, ncorrect = 0, 0
    for sentence, label in wsc_utils.jsonl_iterator(arguments.dataset_path + arguments.split + '.jsonl', eval=True):
        pred = roberta.disambiguate_pronoun(sentence)
        nsamples += 1
        if pred == label:
            ncorrect += 1
    print('Accuracy: ' + str(ncorrect / float(nsamples)))
