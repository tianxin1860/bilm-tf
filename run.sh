export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=0
python setup.py install
python  -m pdb bin/train_elmo.py --train_prefix='./baike/train/sentence_*.txt' --vocab_file ./baike/vocabulary_min5k.txt --save_dir output/checkout/ --dropout 0.1 $@
