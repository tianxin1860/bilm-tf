export CUDA_VISIBLE_DEVICES=0
python -m pdb bin/train_elmo.py --train_prefix='./baike/train/sentence_file_15519.txt' --vocab_file ./baike/vocabulary_min10.txt --save_dir output/checkout/
