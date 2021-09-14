from conllu import parse
import itertools
import random

# ------------------------------------- functions --------------------------------------

def ud_permute(ud_data, sentence_len_limit=10, no_sentences = 100,
               shuffle_level='n1', permutation_type='linear', permutation_limit=None):
    # parse data
    sentences = parse(ud_data)
    # prep
    all_permuted_sentences = []
    # iterate over all sentences
    for sentence in sentences:
        #limit sentence len and no sents
        if len(sentence) < sentence_len_limit and len(all_permuted_sentences) < no_sentences:
            print(len(sentence))
            print(sentence, 'sentence')
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
        return all_permuted_sentences








