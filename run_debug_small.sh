export CUDA_VISIBLE_DEVICES=0
python setup.py install
python  bin/train_elmo.py \
	--train_prefix='./baike/train/sentence_file_19199.txt' \
	--vocab_file ./baike/vocabulary_min5k.txt \
	--save_dir output/checkout/ \
	--para_print \
	--log_interval 1 \
	--gpu_num 1 \
	--batch_size 2 \
	--projection_dim 2 \
	--lstm_dim 3 \
	--learning_rate 1000 \
	--detail \
	--n_layers 2 \
	--n_steps 5 \
	--load_dir output/ckpt/model.ckpt-0 \
	$@
