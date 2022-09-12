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
    def __init__(self, corpus_file):
        self.corpus_file = corpus_file

    def __iter__(self):
        from gensim.utils import tokenize
        from gensim import utils
        with utils.open(self.corpus_file, 'r', encoding='utf-8') as fin:
            for line in fin:# for line in open(os.path.join(self.corpus_path, fname), "r"):
                yield line.split()#     yield line.split()


def train(corpus_file, model_name, size=150, min_count=2, window=3, epoch=5):
    import logging

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = FastText(vector_size=size, min_count=min_count, window=window)
    model.build_vocab(corpus_iterable=MyCorpus(corpus_file))
    total_example = model.corpus_count
    model.train(corpus_iterable=MyCorpus(corpus_file), total_examples=total_example, epochs=epoch)
    model.save(model_name)
    os.remove(corpus_file)


if __name__ == '__main__':
    import argparse
    import sys
    import time
    import datetime
    # import the pkg
    sys.path.extend(["../../"])
    import fasttext_chunk.train_vsim.log_v1 as log
    # parser
    parser = argparse.ArgumentParser(description='from dataset get corpus and model')
    parser.add_argument('--corpus_dir', type=str, help=" the dataset path")
    parser.add_argument('--average_size', type=int, default=4096, help="the chunk's average size (unit: B, default:4096)")
    parser.add_argument('--model', type=str, help="result model path(include name)")
    parser.add_argument('--package_count', type=int, default=4, help="words of each sentences")
    parser.add_argument('--feature_count', type=int, default=12, help="block's feature count, default(12)")
    parser.add_argument('--model_dimension', type=int, default=150, help="model_dimension(default:150)")
    parser.add_argument('--min_count', type=int, default=2, help="model's min_count")
    parser.add_argument('--window', type=int, default=3, help="model's window")
    parser.add_argument('--epoch', type=int, default=5, help="model's epoch iterator")
    args = parser.parse_args()

    # get params
    corpus_dir = args.corpus_dir
    train_file = corpus_dir + "_learning/learning_file.txt"
    average_size = args.average_size
    model = args.model
    feature_count = args.feature_count
    model_dimension = args.model_dimension
    min_count = args.min_count
    window = args.window
    epoch = args.epoch
    package_count = args.package_count

    # log
    l = log.LogSystem()
    l.print_log("corpus_dir : %s" % corpus_dir)
    l.print_log("train_dir : %s" % train_file)
    l.print_log("average_size : %d" % average_size)
    l.print_log("model : %s" % model)
    l.print_log("words of each sentences : %d" % package_count)
    l.print_log("feature_count : %d" % feature_count)
    l.print_log("model_dimension : %d" % model_dimension)
    l.print_log("min_count : %d" % min_count)
    l.print_log("window : %d" % window)
    l.print_log("epoch : %d" % epoch)

    # get corpus
    from fasttext_chunk.train_vsim.get_corpus_v3 import get_corpus
    start = time.clock()
    get_corpus(corpus_dir, package_count=package_count, avg_size=average_size, feature_count=feature_count)
    train_time = time.clock()
    l.print_log("get corpus time : %s" % (train_time - start))
    log_month = datetime.datetime.now().month
    log_day = datetime.datetime.now().day
    with open(str(log_month) + "-" + str(log_day), "a+") as log_file:
        log_file.write("\n" + "="*30+"\n get corpus time : %s \n corpus average size : %s" % (train_time - start, average_size))
    # train
    train(train_file, model, window=window, min_count=min_count, size=model_dimension,epoch=epoch)
    end = time.clock()
    l.print_log("get model time : %s" % (end-train_time))
    with open(str(log_month) + "-" + str(log_day), "a+") as log_file:
        log_file.write("\n get model time : %s" % (end-train_time)
                       + "\n average_size : %d" % average_size
                       + "\n model : %s" % model
                       + "\n feature_count : %d" % feature_count
                       + "\n model_dimension : %d" % model_dimension
                       + "\n min_count : %d" % min_count
                       + "\n window : %d" % window
                       + "\n epoch : %d" % epoch
                       + "\n" + "=" * 30)
