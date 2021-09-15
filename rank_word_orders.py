from roberta.helpers import load_shuffled_model
from fairseq import utils
import torch.nn.functional as F
import argparse
import torch
from statistics import mean
from fairseq.data import Dictionary
import numpy as np
import copy
from utils.rand_word_order_utils import ud_permute
from scipy.stats import spearmanr

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
        reduction='none',
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
    mask_idx = dictionary.add_symbol("<mask>")

    all_sent_ppl = []

    for sent_idx, sentence in enumerate(sentences):
        with torch.no_grad():
            print(sentence)
            tokens = roberta.encode(sentence)
            sents_all_tokens_masked = []
            all_token_masks_idxs = []

            for token_to_mask_idx, _ in enumerate(tokens):
                if token_to_mask_idx != 0 and token_to_mask_idx != len(tokens) - 1:
                    new_tokens = tokens.clone()
                    new_tokens[token_to_mask_idx] = mask_idx
                    sents_all_tokens_masked.append(new_tokens)
                    # add token_to_mask_idx *  (len(tokens) - 2) to get index right
                    all_token_masks_idxs.append(token_to_mask_idx)

            sents_all_tokens_masked = torch.stack(sents_all_tokens_masked).to(device)
            features = roberta.model(src_tokens=sents_all_tokens_masked, device=device)
            logits = features[0].squeeze() #, token_to_mask_idx, :].squeeze()
            # calc loss
            # reshape lm_logits from (N,T,C) to (N*T,C) and repeat targets along batch dim
            lm_logits = logits.view(-1, logits.size(-1))
            lm_targets =  tokens.repeat(logits.shape[0],1).to(device)
            lm_targets = lm_targets.view(-1)
            # compute cross entropy
            lm_loss_all = compute_cross_entropy_loss(lm_logits, lm_targets, dictionary.pad())#.item()
            # reshape to original shape
            lm_loss_all = lm_loss_all.reshape(logits.shape[0], logits.shape[1])
            #iterate over sents extract relevant losses
            lm_loss_target = []
            for s_idx in range(lm_loss_all.shape[0]):
                lm_loss_target.append(lm_loss_all[s_idx, all_token_masks_idxs[s_idx]].item())

            #divide by sent len / take mean
            #lm_loss_target  = [l / len(tokens) - 2 for l in lm_loss_target]
            sent_ppl = np.exp(np.mean(lm_loss_target))
            print(sent_ppl, ' :per sent. perplexity')
            all_sent_ppl.append(sent_ppl)
    mean_ppl = mean(all_sent_ppl)
    print(mean_ppl, " :all_sent_mean_ppl")
    return mean_ppl, all_sent_ppl

def main():
    parser = argparse.ArgumentParser(description="generate token embeddings from corpus");
    parser.add_argument('-d', "--dataset_path", type=str);
    parser.add_argument('-m', "--model_path", type=str);
    parser.add_argument('-l', "--max_sentence_len", type=int, default=10);
    parser.add_argument('-no', "--no_sentences", type=int, default=100);

    arguments = parser.parse_args();

    # load dataset
    dataset_file = open(arguments.dataset_path, 'r').read()
    # pass to shuffle function, returns list of lists where inner list is of all perms per sentence
    sentence_permutations, leven_distances_to_orig = ud_permute(dataset_file, no_sentences=arguments.no_sentences,
                                       sentence_len_limit=arguments.max_sentence_len)
    print(len(sentence_permutations), ' no sents')
    # flatten list for now since we just compute a final perp score and turn each sublist into a string
    sentences = [' '.join(sent_list) for sublist in sentence_permutations for sent_list in sublist]
    print(len(sentences), ' no sents flattened')
    #get ppl
    mean_ppl, all_sent_ppl = compute_perplexity(arguments, sentences)
    #compute correlation between ppl and levenstein distance
    corr = spearmanr(all_sent_ppl, leven_distances_to_orig)
    print(corr, " :correlation of perplexity to leven distance to orig order.")




if __name__ == '__main__':
    main();




