import os
import sys
sys.path.append('../')
import torch
from torch import nn
from torch.nn import functional as F

from fasttext.data_prepare import get_minhash_from_chunk
from fasttext.utils import get_n_grams
import pickle
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Fasttext(nn.Module):
  def __init__(self, vocab_size, ngram_size, hidden):
    super(Fasttext, self).__init__()
    self.vocab_size = vocab_size
    self.ngram_size = ngram_size
    self.hidden = hidden

    # we use two embedding between input word and other words in window
    self.in_embedding = nn.Embedding(self.vocab_size, self.hidden)
    self.ngram_embedding = nn.Embedding(self.ngram_size, self.hidden)
    self.out_embedding = nn.Embedding(self.vocab_size, self.hidden)
    self.word2idx = dict()
    self.ngram2idx = dict()

  # 最后如何进行预测？ ngram 还是要的
  # input_embedding 一个 minhash对应一个，取平均，  ngram 也是
  # 所有的平均加起来作为改数据块的向量
  def forward(self, input_labels, pos_labels, neg_labels, ngram_labels):
    input_embedding = self.in_embedding(input_labels) # [batch, hidden]
    ngram_embedding = self.ngram_embedding(ngram_labels)  # [batch, xx, hidden]
    ngram_embedding = torch.mean(ngram_embedding, dim=1) # [batch, hidden]
    input_embedding = torch.add(input_embedding, ngram_embedding) / 2
    pos_embedding = self.out_embedding(pos_labels) # [batch, window * 2, hidden]
    neg_embedding = self.out_embedding(neg_labels) # [batch, window * 2 * k, hidden]

    input_embedding = input_embedding.unsqueeze(2) # [batch, hidden, 1] must be the same dimension when use torch.bmm

    pos_dot = torch.bmm(pos_embedding, input_embedding) # [batch, window * 2, 1]
    neg_dot = torch.bmm(neg_embedding, -input_embedding) # [batch, window * 2 * k, 1]

    pos_dot = pos_dot.squeeze(2) # [batch, window * 2]
    neg_dot = neg_dot.squeeze(2) # [batch, window * 2 * k]

    pos_loss = F.logsigmoid(pos_dot).sum(1)
    neg_loss = F.logsigmoid(neg_dot).sum(1)

    loss = neg_loss + pos_loss

    return -loss

  def get_input_embedding(self):
    # get weights to build an application for evaluation
    return self.in_embedding.weight.detach()
  def get_ngram_embedding(self):
    # get weights to build an application for evaluation
    return self.ngram_embedding.weight.detach()
  def set_index_file(self, word2idx, ngram2idx):
    self.word2idx = word2idx
    self.ngram2idx = ngram2idx

  def save_index_file(self, path="."):
    with open(os.path.join(path, "word2idx.pkl"), "wb") as writer:
      pickle.dump(self.word2idx, writer)
    with open(os.path.join(path, "ngram2idx.pkl"), "wb") as writer:
      pickle.dump(self.ngram2idx, writer)

  def predict(self, data, sliding_window, num_perm, ngram_window, shingles):
    """
    :param ngram_window:
    :param num_perm:
    :param sliding_window:
    :param data: 数据类型应该是list, 为处理之后的数据？处理之后吧
    :return: 该数据块的向量
    """

    #  首先是获取ngram，以及对应的向量
    _, minhash_digest_base64 = get_minhash_from_chunk(chunk_data=data, sliding_window=sliding_window, num_perm=num_perm, shingles=shingles)
    # print(f"dic: {self.ngram2idx}")
    ngrams = get_n_grams(minhash_digest_base64, n=ngram_window)
    ngrams2idx = [self.ngram2idx.get(ngram, self.ngram2idx['<UNK>']) for ngram in ngrams]
    ngrams2idx = torch.LongTensor(ngrams2idx).long().to(device)
    ngram_all = self.ngram_embedding.weight.to(device)[ngrams2idx, :]
    # ngram_all = self.ngram_embedding.weight.detach()[ngrams2idx, :]
    ngram_avg = torch.mean(ngram_all, dim=0)

    # 获得所有的词向量
    words2idx = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in minhash_digest_base64]
    words2idx = torch.LongTensor(words2idx).long().to(device)
    word_all = self.in_embedding.weight.to(device)[words2idx, :]
    word_avg = torch.mean(word_all, dim=0)

    chunk2vec = torch.add(ngram_avg, word_avg) / 2
    return chunk2vec





