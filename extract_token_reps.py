from roberta.helpers import load_shuffled_model
from fairseq.models.roberta import RobertaModel
from roberta.dataset import get_dataset
import argparse
from collections import defaultdict
import json


def main():
    parser = argparse.ArgumentParser(description="generate token embeddings from corpus");
    parser.add_argument('-d', "--dataset_name", type=str, default='ptb');
    parser.add_argument('-m', "--model_path", type=str);
    parser.add_argument('-o', "--out_folder", type=str);
    parser.add_argument('-l', "--context_len", type=int, default=100);
    parser.add_argument('-b', "--batch_size", type=int, default=64);
    parser.add_argument('-no', "--no_contexts_limit", type=int, default=100);

    arguments = parser.parse_args();

    # load dataset
    train_iter, val_iter, test_iter  = get_dataset(arguments.dataset_name)

    # Load pre-trained model (weights) and extract reps
    roberta = load_shuffled_model(arguments.model_path)
    # make default dictionary for storing extracted embeddings
    embed_dict = defaultdict(list)
    # iterate over train set and extract features
    for line in train_iter:
        if len(line.strip()) > 0:
            try:
                enc = roberta.extract_features_aligned_to_words(line.strip())
                for tok in enc:
                    if len(embed_dict[str(tok)]) < arguments.no_contexts_limit:
                        print('{:100}{} (...)'.format(str(tok), tok.vector[-1:].cpu().detach().numpy()))
                        embed_dict[str(tok)].append(tok.vector.cpu().detach().numpy())
            except:
                continue

    # write out embeds file
    out_file_name = arguments.out_folder + 'embs-' + arguments.dataset_name + \
                    '-cntx_count-' + str(arguments.no_contexts_limit) + '.txt'
    json.dump(embed_dict, out_file_name)

if __name__ == '__main__':
    main();




