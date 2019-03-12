#!/bin/bash

log=$1
cat ${log} | grep "Batch" | grep "train_perplexity=" | cut -f2 -d"," | cut -f2 -d"="

