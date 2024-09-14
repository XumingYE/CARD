import torch

from card.utils import get_n_grams
from torch.utils import data as tud


class EmbeddingDataset(tud.Dataset):
  def __init__(self, text, word2idx, word_freqs, ngram2idx, idx2word, n_gram, neighbor, negative_sample):
    super(EmbeddingDataset, self).__init__()
    self.text_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in text] # 全为索引
    self.text_encoded = torch.LongTensor(self.text_encoded)
    self.n_gram = n_gram
    # self.word2ngrams = get_n_grams(text, n=n_gram, horizon=False) # 全分为ngram, 太大了 - 最终是得到该值所在的索引
    self.idx2word = idx2word
    self.word2idx = word2idx
    self.ngram2idx = ngram2idx
    self.word_freqs = torch.Tensor(word_freqs)
    self.neighbor = neighbor
    self.negative_sample = negative_sample

  def __len__(self):
    return len(self.text_encoded)

  def __getitem__(self, idx):
    center_word = self.text_encoded[idx]
    # print("word index: {}".format(center_word))
    word = self.idx2word[center_word.item()]
    # print("word: {}".format(word))
    ngram_set = get_n_grams(word, n=self.n_gram, horizon=False, single_word=True)
    # print("ngram_set: {}".format(ngram_set))

    if len(ngram_set) > 12:
      while 'UNK' in ngram_set:
        ngram_set.remove('UNK')

    if len(ngram_set) < 12:
      for i in range(12 - len(ngram_set)):
        ngram_set.append("UNK")

    if len(ngram_set) > 12:
      ngram_set = ngram_set[:12]

    ngram_words = [self.ngram2idx.get(ngram, self.ngram2idx['<UNK>']) for ngram in ngram_set]
    # print("ngram_words: {}".format(ngram_words))
    ngram_words = torch.LongTensor(ngram_words)
    # print("ngram_words index : {}".format(ngram_words))
    # get words in window exception center word
    pos_idx = [i for i in range(idx - self.neighbor, idx)] + [i for i in range(idx + 1, idx + self.neighbor + 1)]
    pos_idx = [i % len(self.text_encoded) for i in pos_idx]

    pos_words = self.text_encoded[pos_idx]

    neg_mask = torch.Tensor(self.word_freqs.clone())
    neg_mask[pos_words] = 0

    neg_words = torch.multinomial(neg_mask, self.negative_sample * pos_words.shape[0], True)
    # check if negative sample failure exists
    if len(set(pos_words.numpy().tolist()) & set(neg_words.numpy().tolist())) > 0:
        print('Need to resample.')

    return center_word, pos_words, neg_words, ngram_words