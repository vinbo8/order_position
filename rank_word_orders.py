from roberta.helpers import load_shuffled_model
from fairseq import utils
import torch.nn.functional as F
import argparse
import torch
from statistics import mean
from fairseq.data import Dictionary
import numpy as np
import copy

sentences = ['I am the walrus.', 'He is superman.']

def compute_cross_entropy_loss(logits, targets, ignore_index=-100):
    """
    Function to compute the cross entropy loss. The default value of
    ignore_index is the same as the default value for F.cross_entropy in
    pytorch.
    """
    assert logits.size(0) == targets.size(
        -1
    ), "Logits and Targets tensor shapes don't match up"

    loss = F.nll_loss(
        F.log_softmax(logits, -1, dtype=torch.float32),
        targets,
        reduction="sum",
        ignore_index=ignore_index,
    )
    return loss

def compute_perplexity(args, sentences):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load pre-trained model (weights) and extract reps
    roberta = load_shuffled_model(args.model_path)
    roberta.eval()
    stride = 512
    #load dict
    dictionary = Dictionary.load(args.model_path + "/dict.txt")
    #mask_idx = dictionary.add_symbol("<mask>")

    all_sent_ppl = []
    for sentence in sentences:
        with torch.no_grad():
            print(sentence)
            tokens = roberta.encode(sentence)
            print(tokens.shape, ' tokens')

            for token_to_mask_idx, _ in enumerate(tokens):
                if token_to_mask_idx != 0 and token_to_mask_idx != len(tokens) - 1:
                    new_tokens = tokens.clone()
                    #new_tokens[token_to_mask_idx] = mask_idx
                    print(new_tokens, ' new_tokens')
                    #print(mask_idx, 'mask_idx')
                    features, _ = roberta.model(src_tokens=tokens)
                    logits = features[0, token_to_mask_idx, :].squeeze()
                    print(logits)
                    #calc loss
                    # reshape lm_logits from (N,T,C) to (N*T,C)
                    lm_logits = logits.view(-1, logits.size(-1))
                    lm_targets = tokens.view(-1)
                    lm_loss = compute_cross_entropy_loss(lm_logits, lm_targets, dictionary.padding_idx)

                    ppl = np.exp(lm_loss)
                    all_sent_ppl.append(ppl)
    mean_sent_ppl = mean(all_sent_ppl)
    print(mean_sent_ppl, "mean_sent_ppl")
    return mean_sent_ppl

def main():
    parser = argparse.ArgumentParser(description="generate token embeddings from corpus");
    parser.add_argument('-d', "--dataset_name", type=str, default='UD');
    parser.add_argument('-m', "--model_path", type=str);
    parser.add_argument('-l', "--max_sentence_len", type=int, default=100);
    parser.add_argument('-no', "--no_sentences", type=int, default=100);

    arguments = parser.parse_args();

    # load dataset
    #train_iter, val_iter, test_iter  = get_dataset(arguments.dataset_name)

    #get ppl
    compute_perplexity(arguments, sentences)



if __name__ == '__main__':
    main();




