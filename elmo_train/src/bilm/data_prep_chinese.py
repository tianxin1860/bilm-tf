#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import os
import sys
import pickle


def write_sentences(sentences, filename):
    """
    write a list of sentences into filename
    input:
        sentences: List[List[string]]
        finename: output file name, string
    """
    print('Write {} sentences into {}'.format(len(sentences), filename))
    with open(filename, 'w') as fin:
        for sentence in sentences:
            fin.write(' '.join(sentence) + '\n')


class Vocabulary(object):
    def __init__(self, filenames):
        """
        filenames: list of filenames, including train, test and dev files
        """
        self.filenames = filenames
        for filename in filenames:
            if not os.path.exists(filename):
                raise ValueError('file {} does not exist'.format(filename))

    def iter_over_words(self):
        num_files = len(self.filenames)
        for i, filename in enumerate(self.filenames, 1):
            if i % 100 == 0:
                print('{} of {} files done!'.format(i, num_files))
                sys.stdout.flush()
            with open(filename) as fin:
                for line in fin:
                    for word in line.strip().split():
                        yield word

    def _count(self):
        words = self.iter_over_words()
        counter = Counter()
        for word in words:
            counter[word] += 1
        return counter

    def build_vocab(self, special_tokens=['<S>', '</S>', '<UNK>'],
                    min_count=1):
        self.token2id = {}
        self.id2token = {}
        for i, special_token in enumerate(special_tokens):
            self.token2id[special_token] = i
            self.id2token[i] = special_token
        cnt = self._count()
        for i, (word, c) in enumerate(cnt.most_common(), len(special_tokens)):
            if c >= min_count:
                self.token2id[word] = i
                self.id2token[i] = word
        self.size = len(self.id2token)
        self.num_tokens = sum(cnt.values())

    def dump(self, filename_txt='vocabulary.txt', filename_info='info.dat'):
        with open(filename_txt, 'w') as fin:
            for i in range(self.size):
                fin.write('{}\n'.format(self.id2token[i]))
        print('Dump {} vocabularies into {}, done'
              .format(self.size, filename_txt))
        print('{} tokens in total'.format(self.num_tokens))
        info = {'token2id': self.token2id,
                'id2token': self.id2token,
                'size': self.size,
                'num_tokens': self.num_tokens
                }
        with open(filename_info, 'wb') as fin:
            pickle.dump(info, fin)
        print('dump infomation into {}, done'.format(filename_info))
