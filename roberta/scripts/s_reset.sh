#!/usr/bin/env bash
task=(MNLI QNLI QQP RTE SST-2 CoLA)
classes=(3 2 2 2 2 2 1)
lr=(1e-5 1e-5 1e-5 2e-5 1e-5 1e-5)
bsize=(32 32 32 16 32 16)
update=(123873 33112 113272 2036 20935 5336)
warmup=(7432 1986 28318 122 1256 320)

for i in ${!task[@]}; do
    name="$1.$2.${task[i]}.$3"
    logfile="../logs/reset_pos/$1.$2.${task[i]}.$3.log"
    sbatch -o $logfile -e $logfile -J $name reset.slurm $1 $2 ${task[i]} ${classes[i]} ${lr[i]} ${bsize[i]} ${update[i]} ${warmup[i]} $3
done

