set -eu


DATA_PATH="/home/ssd3/tianxin04/data/baike_mini/"
LD_LIBRARY_PATH="/home/ssd3/tianxin04/share/cudnn-5.1/cuda/lib64:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python  train_elmo_mini.py \
        --train_prefix "${DATA_PATH}/train/sentence_file_*"  \
        --vocab_file "./package/vocabulary_min5k.txt" \
        --save_dir "checkpoints" > log/job.log 2>&1
