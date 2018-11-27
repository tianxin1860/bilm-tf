
import argparse

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file)

    # define the options
    batch_size = 128  # batch size for each GPU
    n_gpus = 8

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = 867616283

    options = {
     'bidirectional': True,
     'dropout': 0.1,
    
     'lstm': {
      'cell_clip': 3,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': 512,
      'use_skip_connections': True},
    
     'all_clip_norm_val': 10.0,
    
     'n_epochs': 10,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 20,
     'n_negative_samples_batch': 8000,
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir, args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--para_print', action='store_true')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--detail', action='store_true')

    args = parser.parse_args()
    main(args)

