import numpy as np
import random
from helpers import load_shuffled_model
import tqdm

model = load_shuffled_model('models/roberta.base.orig')
token_bank = list(model.bpe.bpe.decoder.keys())

with open("uniform.bpe", "w") as f:
    for i in tqdm.tqdm([" ".join([str(j) for j in random.choices(token_bank, k=i)])
                        for i in random.choices(range(5, 30), k=100000)]):
        f.write(f"{i}\n")

with open("odd_even.bpe", "w") as f:
    odd_bank = token_bank[:len(token_bank) // 2]
    even_bank = token_bank[len(token_bank) // 2:]
    odd = [" ".join([str(j) for j in random.choice(token_bank, k=i)])
           for i in random.choices(range(5, 30, 2), k=50000)]
    even = [" ".join([str(j) for j in random.choice(token_bank, k=i)])
            for i in random.choices(range(6, 30, 2), k=50000)]
    final = odd + even
    random.shuffle(final)
    for i in final:
        f.write(f"{i}\n")
