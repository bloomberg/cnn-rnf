# Convolutional Neural Networks with Recurrent Neural Filters


Author: Yi Yang

Contact: yyang464@bloomberg.net


## Basic description

This is the Python implementation of the recurrent neural filters for convolutional neural networks, 
described in

    Yi Yang
    "Convolutional Neural Networks with Recurrent Neural Filters"
    EMNLP 2018

[[pdf]](https://arxiv.org/abs/1808.09315)

BibTeX

    @inproceedings{yang2018convolutional,
      title={Convolutional Neural Networks with Recurrent Neural Filters},
      author={Yang, Yi},
      booktitle={Proceedings of Empirical Methods in Natural Language Processing},
      year={2018}
    }

## Dependencies

1. [TensorFlow](https://www.tensorflow.org/)
2. [Keras](https://keras.io/)
3. Optional: [CUDA Toolkit](http://docs.nvidia.com/cuda/) for GPU programming.


## Data

We use the Stanford Sentiment Treebank (SST) datasets processed by Lei et al. (2015). 
Please put all the files of [this directory](https://github.com/taolei87/text_convnet/tree/master/data) into the [data/sst_text_convnet](data/sst_text_convnet) folder.

Please download the pre-trained [GloVe vectors](http://nlp.stanford.edu/data/glove.840B.300d.zip) and unzip it into the [data](data) folder.


## Results

Running the code requires two steps:

1. Prepare the data and generate the required data files
    ```
    # binary sentiment classification
    python proc_data.py data/stsa.binary.pkl

    # fine-grained sentiment classification
    python proc_data.py --train-path data/sst_text_convnet/stsa.fine.phrases.train \
                        --dev-path   data/sst_text_convnet/stsa.fine.dev \
                        --test-path  data/sst_text_convnet/stsa.fine.test \
                        data/stsa.fine.pkl
    ```

2. CNNs for sentiment classification with linear filters and recurrent neural filters (RNFs)
    ```
    # binary sentiment classification
    python cnn_keras.py --filter-type linear data/stsa.binary.pkl
    python cnn_keras.py --filter-type rnf data/stsa.binary.pkl

    # fine-grained sentiment classification
    python cnn_keras.py --filter-type linear data/stsa.fine.pkl
    python cnn_keras.py --filter-type rnf data/stsa.fine.pkl
    ```

Hyperparameter tunning may be needed to achive the best results reported in the paper. 

Unfortunately, I failed to find out how to entirely eliminate randomness for training Keras-based models. 
However, you should be easily able to achieve 89\%+ and 52\%+ accuracies with RNFs after a few runs.

Recurrent neural filters consistently outperform linear filters across different filter widths,
by 3-4\% accuracy.


