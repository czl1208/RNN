from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np
import tensorflow as tf

Py3 = sys.version_info[0] == 3

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", " ").split()
    else:
      return f.read().decode("utf-8").replace("\n", " ").split()

def id_to_word(arr):
  filename='/Users/caozhongli/simple-examples/data/pptx.train.txt'
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  print(len(words))
  return [[words[i - 1] for i in row if i > 0] for row in arr]


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = []
  with tf.gfile.GFile(filename, "r") as f:
    sentences = f.read().split("\n")
    for sentence in sentences:
        line = [word_to_id[word] + 1 for word in sentence.split() if word in word_to_id]
        data.append(line)
  mx = 0;
  for line in data:
      mx = max(len(line), mx)
  data = [line for line in data if len(line) > 0]
  for line in data:
      num = line[len(line) - 1]
      line.pop(len(line) - 1)
      while len(line) < mx:
          line.extend([0])
      line.extend([num])
  return data


def read_raw_data():

  train_path = os.path.join("/Users/caozhongli/simple-examples/data/", "pptx.train.txt")
  #valid_path = os.path.join(data_path, "pptx.train.txt")
  #test_path = os.path.join(data_path, "pptx.train.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  #valid_data, _ = _file_to_word_ids(valid_path, word_to_id)
  #test_data, _ = _file_to_word_ids(test_path, word_to_id)
  #vocabulary = len(word_to_id)
  train_data = np.asarray(train_data)
  print(train_data)
  return train_data, len(word_to_id)

read_raw_data()

