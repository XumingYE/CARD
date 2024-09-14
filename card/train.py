import sys
sys.path.append('../')
import pickle
import torch
from time import process_time
from torch.utils import data as tud
from torch import optim
import numpy as np
from card.dataset import EmbeddingDataset
from model import Fasttext
from card import data_prepare
from card import options
dtype=torch.FloatTensor
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# 创建一个日志记录器
logger = logging.getLogger(__name__)
# Preparation - settings


if __name__ == '__main__':
  
  args = options.args_parser()
  args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logger.info("Current Device:", args.device)
  logger.info("model: {}".format(args.model))

  # 准备语料：
  data_prepare.get_corpus_for_train(args.corpus_dir, avg_size=args.average_size, sliding_window=args.feature_window,
                                    num_perm=args.num_perm)

  text, word2idx, ngram2idx, word_freqs, idx2word = data_prepare.establish_index(
    args.corpus_dir + '_learning_corpus', max_vocab=args.max_vocab, max_ngram=args.max_ngram, ngram_window=args.ngram_window)
  logger.info("Finish Data Preparation")

  dataset = EmbeddingDataset(
        text, word2idx=word2idx, word_freqs=word_freqs, ngram2idx=ngram2idx, idx2word=idx2word,
        n_gram=args.ngram_window,
        neighbor=args.neighbor,
        negative_sample=args.negative_sample
    )
  dataloader = tud.DataLoader(dataset, args.bs, True, num_workers=args.num_workers, pin_memory=True)
  fasttext = Fasttext(args.max_vocab, args.max_ngram, args.hidden).to(args.device)
  fasttext.set_index_file(word2idx=word2idx, ngram2idx=ngram2idx)
  optimizer = optim.Adam(fasttext.parameters(), lr=args.lr)

  logger.info("Step in one epoch:{}".format(len(dataloader)))
  from tqdm import tqdm

  total_time_start = process_time()
  # start = time()
  min_loss = 10000
  temp_loss = 10000
  patience = 3
  counter = 0
  theta = 0.005
  best_model = {}
  best_word2idx = {}
  best_ngram2idx = {}
  for epoch in range(args.epochs):
    for step, (input_label, pos_label, neg_label, ngram_label) in enumerate(tqdm(dataloader)):
      input_label = input_label.long().to(args.device)
      pos_label = pos_label.long().to(args.device)
      neg_label = neg_label.long().to(args.device)
      ngram_label = ngram_label.long().to(args.device)
      # 3 step in torch
      optimizer.zero_grad()
      loss = fasttext(input_label, pos_label, neg_label, ngram_label).mean()
      loss.backward()
      optimizer.step()
      if temp_loss > loss.item():
        temp_loss = loss.item()
        
    if temp_loss < min_loss - theta:
      min_loss = temp_loss
      counter = 0 
      logger.info("New min loss: {}".format(min_loss))
      best_model = fasttext.state_dict()
      best_word2idx = fasttext.word2idx
      best_ngram2idx = fasttext.ngram2idx
    else:
      counter += 1
    
    if counter >= patience:
      logger.info("Early stopping triggered")
      break
  total_time_end = process_time()
  logger.info("total time: {}".format(total_time_end - total_time_start))
  torch.save(best_model, args.model)
  with open(os.path.join(os.path.dirname(args.model), os.path.basename(args.model).split('.')[0] + "word2idx.pkl"), "wb") as writer:
    pickle.dump(best_word2idx, writer)
  with open(os.path.join(os.path.dirname(args.model), os.path.basename(args.model).split('.')[0] + "ngram2idx.pkl"), "wb") as writer:
    pickle.dump(best_ngram2idx, writer)


