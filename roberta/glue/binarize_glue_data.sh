#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
if [[ $# -ne 2 ]]; then
  echo "Run as following:"
  echo "./examples/roberta/preprocess_GLUE_tasks.sh <glud_data_folder> <task_name>"
  exit 1
fi

GLUE_DATA_FOLDER=$1
TASKS=$2 # QQP

if [ "$TASKS" = "ALL" ]
then
  TASKS="QQP MNLI QNLI RTE SST-2 CoLA"
fi

for TASK in $TASKS
do
  echo "Binarizing $TASK"
  TASK_DATA_FOLDER="$GLUE_DATA_FOLDER/$TASK"
  SPLITS="train dev test"
  INPUT_COUNT=2
  if [ "$TASK" = "SST-2" ]
  then
    INPUT_COUNT=1
  elif [ "$TASK" = "CoLA" ]
  then
    INPUT_COUNT=1
  fi
  DEVPREF="$TASK_DATA_FOLDER/processed/dev.LANG"
  TESTPREF="$TASK_DATA_FOLDER/processed/test.LANG"
  if [ "$TASK" = "MNLI" ]
  then
    DEVPREF="$TASK_DATA_FOLDER/processed/dev_matched.LANG,$TASK_DATA_FOLDER/processed/dev_mismatched.LANG"
    TESTPREF="$TASK_DATA_FOLDER/processed/test_matched.LANG,$TASK_DATA_FOLDER/processed/test_mismatched.LANG"
  fi

  # Run fairseq preprocessing:
  for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1)))
  do
    LANG="input$INPUT_TYPE"
    fairseq-preprocess \
      --only-source \
      --trainpref "$TASK_DATA_FOLDER/processed/train.$LANG" \
      --validpref "${DEVPREF//LANG/$LANG}" \
      --testpref "${TESTPREF//LANG/$LANG}" \
      --destdir "$TASK-bin/$LANG" \
      --workers 60 \
      --srcdict dict.txt;
  done
  fairseq-preprocess \
    --only-source \
    --trainpref "$TASK_DATA_FOLDER/processed/train.label" \
    --validpref "${DEVPREF//LANG/label}" \
    --destdir "$TASK-bin/label" \
    --workers 60;
done

