from conllu import parse
import itertools
import random
from Levenshtein import distance as levenshtein_distance
import nltk

# ------------------------------------- functions --------------------------------------

def ud_permute(ud_data, sentence_len_limit=None, no_sentences=None,
               shuffle_level='n1', permutation_type='linear', permutation_limit=None):
    # parse data
    sentences = parse(ud_data)
    # prep
    all_permuted_sentences = []
    leven_distances_to_orig = []
    bleu_to_orig = []

    # iterate over all sentences
    for sentence in sentences:
        #limit sentence len and no sents
        if len(sentence) < sentence_len_limit and len(sentence) > 4 and len(all_permuted_sentences) < no_sentences:
            # just randomly shuffle all toks irrespective of heirarch. structure
            if permutation_type == 'linear':
                # sent to list of tokens
                token_list = [t['form'] for t in sentence]
                # permute
                permutation_list = list(itertools.permutations(token_list))
                # if sub-sampling on
                if permutation_limit:
                    permutation_list = random.sample(permutation_list, permutation_limit)
                all_permuted_sentences.append(permutation_list)
                # compute leven distaces
                for permutation in permutation_list:
                    ld = levenshtein_distance(' '.join(token_list), ' '.join(permutation))
                    leven_distances_to_orig.append(ld)
                    bs = nltk.translate.bleu_score.sentence_bleu(' '.join(token_list), ' '.join(permutation))
                    bleu_to_orig.append(bs)
    return all_permuted_sentences, leven_distances_to_orig, bleu_to_orig








