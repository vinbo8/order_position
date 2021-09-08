import re
import numpy as np
import sys

sys.stdout.write("\t")
for task in ["QNLI", "RTE", "QQP", "SST-2", "MNLI", "CoLA"]:
    sys.stdout.write(f"{task}\t")

sys.stdout.write("\n")
exp = re.compile(r'best_accuracy (.*)')
for embed in ["orig"]:
    for mode in ["all", "ft", "test"]:
        sys.stdout.write(f"roberta.{embed}.{mode}\t")
        for task in ["QNLI", "RTE", "QQP", "SST-2", "MNLI", "CoLA"]:
            current = []
            for seed in [0]:
                with open(f"roberta.base.{embed}.{mode}.{task}.{seed}.log") as f:
                    current.append(float(exp.findall(f.read())[-1]))

            sys.stdout.write(f"{np.mean(current):.2f},{np.std(current):.2f}\t")  
        sys.stdout.write("\n")
