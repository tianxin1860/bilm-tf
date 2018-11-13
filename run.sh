export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python  bin/train_elmo.py --train_prefix='./baike/train/sentence_*.txt' --vocab_file ./baike/vocabulary_min10.txt --save_dir output/checkout/
