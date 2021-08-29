#!/usr/bin/env bash
task=(MNLI QNLI QQP RTE SST-2 MRPC CoLA STS-B)
classes=(3 2 2 2 2 2 2 1)
lr=(1e-5 1e-5 1e-5 2e-5 1e-5 1e-5 1e-5 2e-5)
bsize=(32 32 32 16 32 16 16 16)
update=(123873 33112 113272 2036 20935 2296 5336 3598)
warmup=(7432 1986 28318 122 1256 137 320 214)

for i in ${!task[@]}; do
	logfile=logs/$1.${task[i]}.log
	sbatch -o $logfile -e $logfile -J $logfile finetune.slurm $1 ${task[i]} ${classes[i]} ${lr[i]} \
	${bsize[i]} ${update[i]} ${warmup[i]}
done

