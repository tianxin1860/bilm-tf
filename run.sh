export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=.:$PYTHONPATH
python  bin/train_elmo.py --train_prefix='./baike/train/sentence_*.txt' --vocab_file ./baike/vocabulary_min5k.txt --save_dir output/checkout/ $@
