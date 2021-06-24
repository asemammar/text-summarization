import sys, pickle, numpy as np
from batcher import Vocab
import pickle
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# https://github.com/inzva/Turkish-GloVe
# Follow the instructions and create gensim word2vec file using pretrained Turkish-Glove embeddings
w2v_file = 'path/to/w2v_vectors.txt'
path_to_vocab = '../../data/mlsumtr/vocab'
save_file = 'path/to/glove-tr_embedding_matrix.pk'
embed_size = 300 # in main.py change params["embed_size"] according to this embed_size

vocab = Vocab(path_to_vocab, 50000)

def load_w2v_model(w2v_file):
    print("Loading w2v Model")
    w2v_embeddings = KeyedVectors.load_word2vec_format(w2v_file)
    return w2v_embeddings

if __name__ == "__main__":

    w2v_embeddings = load_w2v_model(w2v_file)
    embedding_matrix = np.zeros((len(vocab.word2id) , embed_size), dtype=np.float32)

    print("glove-tr matrices loaded, producing embedding matrix")

    for word, i in vocab.word2id.items():
        if word in w2v_embeddings.vocab:
            embedding_matrix[i] = w2v_embeddings.get_vector(word)

    print("Embedding matrices successfully created")

    with open(save_file, "wb") as file:
        pickle.dump(embedding_matrix, file)

    print("Embedding matrices successfully saved using pickle")