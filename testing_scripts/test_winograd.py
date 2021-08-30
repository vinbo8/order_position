from roberta.helpers import load_shuffled_model
from fairseq.models.roberta import RobertaModel
from examples.roberta.wsc import wsc_utils  # also loads WSC task and criterion
import torch

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# Load pre-trained model (weights) and test o
#model = load_shuffled_model('roberta/models/roberta.base.orig')
roberta = RobertaModel.from_pretrained('roberta/models/roberta.base.orig', 'model.pt', 'data/wino/WSC/')
roberta.cuda()
nsamples, ncorrect = 0, 0
for sentence, label in wsc_utils.jsonl_iterator('data/wino/WSC/val.jsonl', eval=True):
    pred = roberta.disambiguate_pronoun(sentence)
    nsamples += 1
    if pred == label:
        ncorrect += 1
print('Accuracy: ' + str(ncorrect / float(nsamples)))
