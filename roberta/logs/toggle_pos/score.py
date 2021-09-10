import re
import numpy as np
import sys

exp = re.compile(r'best_accuracy (.*)')
for embed in ["shuffle.n1"]:
    for mode in ["keep", "invert"]:
        sys.stdout.write(f"roberta.{embed}.{mode}\t")
        for task in ["QNLI", "RTE", "QQP", "SST-2", "MNLI", "CoLA"]:
            current = []
            for seed in [0, 42, 100]:
                with open(f"roberta.base.{embed}.{mode}.{task}.{seed}.log") as f:
                    current.append(float(exp.findall(f.read())[-1]))

            sys.stdout.write(f"{np.mean(current):.2f},{np.std(current):.2f}\t")  
        sys.stdout.write("\n")
