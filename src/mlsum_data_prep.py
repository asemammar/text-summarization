# -*- coding: utf-8 -*-
"""mlsum_data_prep.ipynb

Prepares the MLSUM Turkish data to feed into Pointer Generator model.
"""

import nltk
import collections
import multiprocessing
import tensorflow as tf
import os

from datasets import load_dataset

nltk.download('punkt')

data_dir = "data/mlsumtr"
chunked_files_dir = f"{data_dir}/chunked"

if not os.path.exists(chunked_files_dir):
  os.makedirs(chunked_files_dir)

dataset = load_dataset("mlsum", "tu")

dataset = dataset.rename_column('text', 'article')
dataset = dataset.rename_column('summary', 'abstract')

def sentence_split(example):
  """
  Splits the sentences of the abtsract field and add tags to the beginning and end of the sentences.
  """
  sent_text = nltk.sent_tokenize(example['abstract'])
  sent_text = ["<s> " + s + " </s>" for s in sent_text]
  example['abstract'] = " ".join(sent_text)
  return example

for s in ['train', 'test', 'validation']:
  d = dataset[s]
  d = d.map(sentence_split)
  d.set_format(type='numpy', columns=['article', 'abstract'])
  if s == 'validation':
    s = 'val'
  d.export(filename=f'{data_dir}/{s}.tfrecord', format="tfrecord")

def _parse_function(example_proto):
  # Create a description of the features.
  feature_description = {
    'text': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'summary': tf.io.FixedLenFeature([], tf.string, default_value='')
  }
  # Parse the input `tf.Example` proto using the dictionary above.
  parsed_example = tf.io.parse_single_example(example_proto, feature_description)
  return parsed_example

def art_abs_example(article, abstract, record_file):
  """
  Builds a tf.train.Example object from an article and an abstract
  args:	
    article : string bytes 
    abstract : string bytes
  """

  def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
      value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

  with tf.io.TFRecordWriter(record_file) as writer:
    feature = {
    'article': _bytes_feature(article),
    'abstract': _bytes_feature(abstract)
    }

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(tf_example.SerializeToString())


for s in ['train', 'test', 'val']:
  raw_dataset = tf.data.TFRecordDataset([f'{s}.tfrecord'])
  parsed_dataset = raw_dataset.map(_parse_function)
  i = 0
  for raw_record in parsed_dataset:
      article = raw_record["text"].numpy().decode()
      abstract = raw_record["summary"].numpy().decode()
      art_abs_example(article, abstract, f"{chunked_files_dir}/{s}_{str(i).zfill(6)}.tfrecords")

def get_tokens(text):
    res = []
    sent_text = nltk.sent_tokenize(text) # this gives us a list of sentences
    # now loop over each sentence and tokenize it separately
    for sentence in sent_text:
        tokenized_text = nltk.word_tokenize(sentence)
        res += tokenized_text

    return res

def count(data):
  vocab_counter_in = collections.Counter()
  i = 0
  for entry in data:
    # print(entry)
    tokens = get_tokens(entry['article']) + get_tokens(entry['abstract'])
    tokens = [t.strip() for t in tokens] # strip
    tokens = [t for t in tokens if t!=""] # remove empty
    vocab_counter_in.update(tokens)
    i += 1
    if i % 1000 == 0:
      print(i)
  return vocab_counter_in

def count_mul(data):
    vocab_counter_in = collections.Counter()
    label = f"{data['s']}_{str(data['k'])}"
    i = 0
    data = data["entries"]['article'] + data["entries"]['abstract']
    for entry in data:
    # print(entry)
        tokens = get_tokens(entry)
        tokens = [t.strip() for t in tokens] # strip
        tokens = [t for t in tokens if t!=""] # remove empty
        vocab_counter_in.update(tokens)
        i += 1
        if i % 5000 == 0:
            print(f"{label}_{str(i)}")
    return vocab_counter_in

def op_serial():
  global dataset
  vocab_counter = collections.Counter()
  for s in ['train', 'test', 'validation']:
    print(s)
    d = dataset[s]
    vocab_counter += count(d)

  return vocab_counter

def op_parallel(process_count=8, batch_size=10000):
  """Performs count operations in parallel.
  """
  global dataset
  vocab_counter = collections.Counter()

  i = 0
  a_pool = multiprocessing.Pool(process_count)
  entries = []

  for s in ['train', 'test', 'validation']:
      print(s)
      d = dataset[s]
      n = batch_size
      k = 0
      for i in range(0, len(d), n):
          k+=1
          entries.append({"s": s, "k": k, "entries": d[i:i + n]})

  result = a_pool.map(count_mul, entries)
  for c in result:
    vocab_counter += c

  return vocab_counter

vocab_counter = op_serial()
# vocab_counter = op_parallel(32, 10000)

print(f"total token: {sum(vocab_counter.values())} \n unique token: {len(list(vocab_counter))}")

with open(os.path.join(data_dir, "vocab"), 'w') as writer:
    for word, count in vocab_counter.most_common(200000):
        writer.write(word + ' ' + str(count) + '\n')
