#!/bin/bash
set -x
set -e

# echo "running run.sh..."

if [ ! -d log ]; then
    mkdir log
else
    rm -r log/*
fi

if [ ! -d output ]; then
    mkdir output
else
    rm -r output/*
fi

mkdir output/models/
mkdir output/results/


PWD_DIR=`pwd`
PYDIR="python3.5.1"
PYLIB="$PYDIR/lib/python3.5/site-packages/"

export PATH="$PWD_DIR/$PYDIR/bin/:$PWD_DIR/$PYDIR/lib/:$PATH"
export PYTHONPATH="$PWD_DIR/$PYDIR/lib/:$PWD_DIR/$PYLIB:$PYTHONPATH"
export LD_LIBRARY_PATH="/home/work/cuda-8.0/lib64:/home/work/cudnn/cudnn_v5/cuda/lib64:/home/work/cuda-8.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH"

echo `which python3` > $PWD_DIR/log/info.log
echo $PATH >> $PWD_DIR/log/info.log 
# ls /home/work/cuda-8.0/lib64 >> $PWD_DIR/log/info.log
# ls /home/work/cudnn/cudnn_v5/cuda/lib64 >> $PWD_DIR/log/info.log
# ls /home/work/cuda-8.0/extras/CUPTI/lib64 >> $PWD_DIR/log/info.log

python -c 'import sys; print(sys.path)' >> $PWD_DIR/log/info.log
echo $WORK_DIR >> $PWD_DIR/log/info.log

DATA_PKG=baike
python3 src/train_elmo_mini.py \
        --save_dir "$PWD_DIR/output/models" \
        --vocab_file "$PWD_DIR/vocabulary.txt" \
        --train_prefix "$DATA_PKG/train/sentence*" 1>$PWD_DIR/log/train.log 2>&1
# dump rnn and embedding weights to hdf5 files
python3 src/dump_weights.py \
        --save_dir "$PWD_DIR/output/models" \
        --outfile "$PWD_DIR/output/results/rnn_weights.hdf5"

python3 src/dump_embedding_weights.py \
        --save_dir "$PWD_DIR/output/models" \
        --outfile "$PWD_DIR/output/results/embedding_weights.hdf5"

python3 src/run_test.py \
        --save_dir "$PWD_DIR/output/models" \
        --vocab_file "$PWD_DIR/vocabulary.txt" \
        --test_prefix "$DATA_PKG/dev/sentence*" \
        --batch_size 256 1>$PWD_DIR/log/test.log 2>&1
