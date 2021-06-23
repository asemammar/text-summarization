import streamlit as st
import tensorflow as tf
import uuid
import os

from train_test_eval import test_model, beam_decode, test_and_serve
from batcher import batcher

from datasets import Dataset

models = {
  "CNNDM - Base": {
    "name": "CNNDM - Base",
    "key": "base",
    "data": "cnndm",
    "embed_size" : 128
  },
  "CNNDM - Word2Vec": {
    "name": "CNNDM - Word2Vec",
    "key": "w2v",
    "data": "cnndm",
    "embed_size" : 300
  },
  "CNNDM - GloVe": {
    "name": "CNNDM - GloVe",
    "key": "glove",
    "data": "cnndm",
    "embed_size" : 300
  },
  "CNNDM - USE": {
    "name": "CNNDM - USE",
    "key": "use",
    "data": "cnndm",
    "embed_size" : 0
  },
  "MLSUM/TR - Base": {
    "name": "MLSUM/TR - Base",
    "key": "base",
    "data": "mlsumtr",
    "embed_size" : 128
  },
  "MLSUM/TR - Word2Vec": {
    "name": "MLSUM/TR - Word2Vec",
    "key": "w2v",
    "data": "mlsumtr",
    "embed_size" : 300
  },
  "MLSUM/TR - GloVe": {
    "name": "MLSUM/TR - GloVe",
    "key": "glove",
    "data": "mlsumtr",
    "embed_size" : 300
  },
  "MLSUM/TR - USE": {
    "name": "MLSUM/TR - USE",
    "key": "use",
    "data": "mlsumtr",
    "embed_size" : 0
  }
}

default_params = {
    'max_enc_len': 400,
    'max_dec_len': 100,
    'max_dec_steps': 120,
    'min_dec_steps': 30,
    'batch_size': 4,
    'beam_size': 4,
    'vocab_size': 50000,
    'embed_size': 300,
    'enc_units': 256,
    'dec_units': 256,
    'attn_units': 512,
    'learning_rate': 0.15,
    'adagrad_init_acc': 0.1,
    'max_grad_norm': 0.8,
    'checkpoints_save_steps': 5000,
    'max_steps': 38000,
    'num_to_test': 5,
    'max_num_to_eval': 100,
    'mode': 'test_request',
    'model_path': None,
    'checkpoint_dir': '../checkpoints/w2v/',
    'test_save_dir': '../test_dir/',
    'data_dir': '../data/cnndm/',
    'vocab_path': '../data/cnndm/vocab',
    'log_file': '',
    'pt_embedding': '',
}

def get_params(model):
  params = default_params.copy()
  model_params = models[model]
  params.update(
    {
      'embed_size': model_params["embed_size"],
      'checkpoint_dir': f'../checkpoints/{model_params["key"]}/',
      'vocab_path': f'../data/{model_params["data"]}/vocab',
    }
  )
  return params

def generate_tfrecords(article, abstract, record_file):
  
  d = Dataset.from_dict({
    "article": [article, article, article, article], #'null', 'null', 'null'],
    "abstract": [abstract,abstract, abstract, abstract] # 'null', 'null', 'null']
  })

  d.set_format(type='numpy', columns=['article', 'abstract'])

  d.export(filename=record_file[:-1], format="tfrecord")

  os.rename(record_file[:-1], record_file)

  return
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

def generate_summary(text, model_str):
  global params
  uid = uuid.uuid4().hex
  filename = f"../temp/{uid}.tfrecords"
  generate_tfrecords(text, "", filename)
  # filename = "../data/cnndm/test_002.tfrecords"
  
  params_ = get_params(model_str)
  params_['data_dir'] = filename
  res = test_and_serve(params_)
  return res


st.title("Summarize Text")
sentence = st.text_area('Please paste your article :', height=30)
model = st.selectbox('Select Model: ', options=list(models.keys()))
button = st.button("Summarize")

with st.spinner("Generating Summary.."):
    if button and sentence:
        text = generate_summary(sentence, model)
        
        st.subheader("Generated Summary:")
        st.markdown(f"Selected Model: __{model}__ ")
        st.info(text)