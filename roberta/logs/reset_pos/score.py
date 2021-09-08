import re
import numpy as np
import sys

exp = re.compile(r'best_accuracy (.*)')
for embed in ["shuffle.n1"]:
    sys.stdout.write(f"roberta.{embed}.reset\t")
    for task in ["QNLI", "RTE", "QQP", "SST-2", "MNLI", "CoLA"]:
        current = []
        for seed in [0, 42, 100]:
            with open(f"roberta.base.{embed}.reset.{task}.{seed}.log") as f:
                current.append(float(exp.findall(f.read())[-1]))

        sys.stdout.write(f"{np.mean(current):.2f},{np.std(current):.2f}\t")  
    sys.stdout.write("\n")
