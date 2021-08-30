from torchtext.datasets import WikiText2, PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_dataset(dataset_name):
    if dataset_name == 'wiki2':
        train_iter = WikiText2(split='train')
    elif dataset_name == 'ptb':
        train_iter = PennTreebank(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    #def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    #    """Converts raw text into a flat Tensor."""
    #    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    #     return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    #train_data = data_lmprocess(train_iter)
    #val_data = data_process(val_iter)
    #test_data = data_process(test_iter)

    return train_iter, val_iter, test_iter






