# Copyright 2018 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import cPickle
import argparse, logging

np.random.seed(1332)

logger = logging.getLogger("cnn_rnf.proc_data")

def build_data(fnames):
    """
    Load and process data.
    """
    revs = []
    vocab = set()
    corpora = []
    for i in xrange(len(fnames)):
        corpora.append(get_corpus(fnames[i]))
    max_l = 0
    for i, corpus in enumerate(corpora):
        for [label, words] in corpus:
            for word in words:
                vocab.add(word)
            datum  = {'y':label,
                      'words': words,
                      'num_words': len(words),
                      'split': i}
            max_l = max(max_l, datum['num_words'])
            revs.append(datum)
    logger.info("vocab size: %d, max sentence length: %d" %(len(vocab), max_l))
    return revs, vocab, max_l
   
def get_corpus(fname):
    corpus = []
    with open(fname, 'rb') as f:
        for line in f:
            line = line.strip()
            line = line.replace("-lrb-", "(")
            line = line.replace("-rrb-", ")")
            parts = line.split()
            corpus.append((int(parts[0]), parts[1:]))
    return corpus


class WordVecs(object):
    """
    Manage precompute embeddings
    """
    def __init__(self, fname, vocab, random=True):
        word_vecs, self.k = self.load_txt_vec(fname, vocab)
        self.random = random
        self.add_unknown_words(word_vecs, vocab)
        self.W, self.word_idx_map = self.get_W(word_vecs)

    def get_W(self, word_vecs):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        vocab_size = len(word_vecs)
        word_idx_map = dict()
        W = np.zeros(shape=(vocab_size+1, self.k))            
        W[0] = np.zeros(self.k)
        i = 1
        for word in word_vecs:
            W[i] = word_vecs[word]
            word_idx_map[word] = i
            i += 1
        return W, word_idx_map
   
    def load_txt_vec(self, fname, vocab):
        word_vecs = {}
        with open(fname, "rb") as f:
            header = f.readline()
            parts = header.strip().split()
            if len(parts) == 2:
                vocab_size, word_dim = map(int, parts)
            else:
                word = parts[0]
                if word in vocab:
                   word_vecs[word] = np.asarray(map(float, parts[1:]))
                word_dim = len(parts) - 1
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                if word in vocab:
                   word_vecs[word] = np.asarray(map(float, parts[1:]))
        logger.info("num words already in word2vec: " + str(len(word_vecs)))
        return word_vecs, word_dim 

    def add_unknown_words(self, word_vecs, vocab):
        """
        0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        """
        for word in vocab:
            if word not in word_vecs: 
                if self.random:
                    word_vecs[word] = np.random.uniform(-0.25, 0.25, self.k)  
                else:
                    word_vecs[word] = np.zeros(self.k)


def parse_args():
    parser = argparse.ArgumentParser(description='VAEs')
    parser.add_argument('--train-path', type=str, default='data/sst_text_convnet/stsa.binary.phrases.train', help="path for training data")
    parser.add_argument('--dev-path', type=str, default='data/sst_text_convnet/stsa.binary.dev', help="path for development data")
    parser.add_argument('--test-path', type=str, default='data/sst_text_convnet/stsa.binary.test', help="path for test data")
    parser.add_argument('--emb-path', type=str, default='data/glove.840B.300d.txt', help="path for pretrained glove embbeddings")
    parser.add_argument('output', type=str, help="path for output pickle file")
    args = parser.parse_args()
    return args
 
def main():
    args = parse_args()
    revs, vocab, max_l = build_data([args.train_path, args.dev_path, args.test_path])
    logger.info("loading and processing pretrained word vectors")
    wordvecs = WordVecs(args.emb_path, vocab)
    cPickle.dump([revs, wordvecs, max_l], open(args.output, 'wb'))
    logger.info("dataset created!")

if __name__=="__main__":    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')
    main()
    logger.info("end logging")
