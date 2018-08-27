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

import cPickle, logging, argparse
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.engine import Layer
from keras.layers import *
from keras.layers.core import *
from keras.layers.embeddings import *
from keras.layers.convolutional import *
from keras.utils import np_utils

from sklearn.metrics import accuracy_score
from proc_data import WordVecs

logger = logging.getLogger("cnn_rnf.cnn_keras")

class ConvInputLayer(Layer):
    """
    Distribute word vectors into chunks - input for the convolution operation
    Input dim: [batch_size x sentence_len x word_vec_dim]
    Output dim: [batch_size x (sentence_len - filter_width + 1) x filter_width x word_vec_dim]
    """
    def __init__(self, filter_width, sent_len, **kwargs):
        super(ConvInputLayer, self).__init__(**kwargs)
        self.filter_width = filter_width
        self.sent_len = sent_len

    def call(self, x):
        chunks = []
        for i in xrange(self.sent_len - self.filter_width + 1):
            chunk = x[:, i:i+self.filter_width, :]
            chunk = K.expand_dims(chunk, 1)
            chunks.append(chunk)
        return K.concatenate(chunks, 1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.sent_len - self.filter_width + 1, self.filter_width, input_shape[-1])


def train_conv_net(datasets,                # word indices of train/dev/test sentences
                   U,                       # pre-trained word embeddings
                   filter_type='linear',    # linear or rnf
                   filter_width=5,          # filter width for n-grams
                   hidden_dim=300,          # dim of sentence vector
                   emb_dropout=0.4,
                   dropout=0.4,
                   recurrent_dropout=0.4, 
                   pool_dropout=0.,
                   batch_size=32,           # mini batch size
                   n_epochs=15):
    """
    train and evaluate convolutional neural network model 
    """
    # print params
    print ("PARAMS: filter_type=%s, filter_width=%d, hidden_dim=%d, emb_dropout=%.2f, dropout=%.2f, recurrent_dropout=%.2f, pool_dropout=%.2f, batch_size=%d"\
               %(filter_type, filter_width, hidden_dim, emb_dropout, dropout, recurrent_dropout, pool_dropout, batch_size))

    # prepare datasets
    train_set, dev_set, test_set = datasets
    train_set_x, dev_set_x, test_set_x = train_set[:,:-1], dev_set[:,:-1], test_set[:,:-1]
    train_set_y, dev_set_y, test_set_y = train_set[:,-1], dev_set[:,-1], test_set[:,-1]
    n_classes = np.max(train_set_y) + 1
    train_set_y = np_utils.to_categorical(train_set_y, n_classes)

    # build model with keras
    n_tok = len(train_set_x[0])
    vocab_size, emb_dim = U.shape
    sequence = Input(shape=(n_tok,), dtype='int32')
    inputs, train_inputs, dev_inputs, test_inputs = [sequence], [train_set_x], [dev_set_x], [test_set_x]

    emb_layer = Embedding(vocab_size, emb_dim, weights=[U], trainable=False, input_length=n_tok)(sequence) 
    emb_layer = Dropout(emb_dropout)(emb_layer)

    if filter_type == 'linear':
        conv_layer = Conv1D(hidden_dim, filter_width, activation='relu')(emb_layer)
    elif filter_type == 'rnf':
        emb_layer = ConvInputLayer(filter_width, n_tok)(emb_layer)
        conv_layer = TimeDistributed(LSTM(hidden_dim, dropout=dropout, recurrent_dropout=recurrent_dropout))(emb_layer)

    text_layer = GlobalMaxPooling1D()(conv_layer)
    text_layer = Dropout(pool_dropout)(text_layer)
    pred_layer = Dense(n_classes, activation='softmax')(text_layer)

    model = Model(inputs=inputs, outputs=pred_layer)
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'])

    # start training
    best_dev_perf, best_test_perf = 0., 0.
    for epo in xrange(n_epochs):
        # training
        model.fit(train_inputs, train_set_y, batch_size=batch_size, epochs=1, verbose=0)

        # evaluation
        dev_pred = model.predict(dev_inputs, batch_size=batch_size, verbose=0).argmax(axis=-1)
        dev_perf = accuracy_score(dev_pred, dev_set_y)
        test_pred = model.predict(test_inputs, batch_size=batch_size, verbose=0).argmax(axis=-1)
        test_perf = accuracy_score(test_pred, test_set_y)
        if dev_perf >= best_dev_perf:
            best_dev_perf, best_test_perf = dev_perf, test_perf
        logger.info("Epoch: %d Dev perf: %.3f Test perf: %.3f" %(epo+1, dev_perf*100, test_perf*100))

    print ("Dev perf: %.3f Test perf: %.3f" %(best_dev_perf*100, best_test_perf*100))

def get_idx_from_sent(words, word_idx_map, max_l=50, filter_width=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = [0] * (filter_width-1)
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l + (filter_width-1)*2:
        x.append(0)
    return x

def make_idx_data(revs, word_idx_map, max_l=50, filter_width=4):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, dev, test = [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['words'], word_idx_map, max_l, filter_width)
        sent.append(rev['y'])
        if rev['split']==0:
            train.append(sent)
        elif rev['split']==1:
            dev.append(sent)
        elif rev['split']==2:
            test.append(sent)
    train = np.array(train,dtype='int32')
    dev = np.array(dev,dtype='int32')
    test = np.array(test,dtype='int32')
    return train, dev, test

def parse_args():
    parser = argparse.ArgumentParser(description="Convolutional neural networks with recurrent neural filters")
    parser.add_argument('--filter-type', type=str, default="linear", choices=['linear', 'rnf'], help="filter type: linear or rnf")
    parser.add_argument('--filter-width', type=int, default=6, help="convolution filter width")
    parser.add_argument('--hidden-dim', type=int, default=300, help="penultimate layer dimension")
    parser.add_argument('--emb-dropout', type=float, default=0.4, help="dropout rate for embedding layer")
    parser.add_argument('--dropout', type=float, default=0.4, help="dropout rate for LSTM linear transformation layer")
    parser.add_argument('--recurrent-dropout', type=float, default=0.4, help="dropout rate for LSTM recurrent layer")
    parser.add_argument('--pool-dropout', type=float, default=0., help="dropout rate for pooling layer")
    parser.add_argument('--batch-size', type=int, default=32, help="mini-batch size")
    parser.add_argument('dataset', type=str, help="processed dataset file path")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    logger.info("loading data...")
    revs, wordvecs, max_l = cPickle.load(open(args.dataset,'rb'))
    logger.info("data loaded!")

    datasets = make_idx_data(revs, wordvecs.word_idx_map, max_l=max_l, filter_width=args.filter_width)
    train_conv_net(datasets, 
                   wordvecs.W, 
                   filter_type=args.filter_type, 
                   filter_width=args.filter_width, 
                   hidden_dim=args.hidden_dim, 
                   emb_dropout=args.emb_dropout, 
                   dropout=args.dropout,
                   recurrent_dropout=args.recurrent_dropout, 
                   pool_dropout=args.pool_dropout, 
                   batch_size=args.batch_size, 
                   n_epochs=20)

if __name__=="__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')
    main()
    logger.info("end logging")
