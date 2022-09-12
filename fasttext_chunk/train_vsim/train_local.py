import os
from gensim.models import FastText
"""
    version: v1
    product data : 2020/10/03 20:22
    :introduction
        Dataset - > Model
    author:
        Mr.Ye
"""


class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path

    def __iter__(self):
        for fname in os.listdir(self.corpus_path):
            for line in open(os.path.join(self.corpus_path, fname), "r"):
                yield line.split()


def train(dir_name, model_name, size=150, min_count=2, window=3):
    """
        V1 gensim 3.8.3
    """
    # sentences = MyCorpus(dir_name)
    # model_ted = FastText(sentences, size=size, min_count=min_count, window=window, workers=6)
    # model_ted.train()
    # model_ted.save(model_name)

    """
        V2 gensim 4.0.0 beta
    """
    model_ted = FastText(corpus_file=dir_name, vector_size=size, min_count=min_count, window=window, workers=6)
    total_words = model_ted.corpus_total_words
    model_ted.train(corpus_file=dir_name, total_words=total_words, epochs=5)

if __name__ == '__main__':
    import argparse
    import sys
    import time
    # import the pkg
    sys.path.extend(["../../"])
    import fasttext_chunk.train_vsim.log_v1 as log
    # get corpus
    from fasttext_chunk.train_vsim.get_corpus_v2 import get_corpus
    start = time.clock()
    get_corpus("/home/ubuntu/Downloads/train_kernel", avg_size=4096, package_count=12)
    # train
    end = time.clock()
    print(str(end-start))
