import sys, pickle, numpy as np
from batcher import Vocab
import pickle
from gensim.models import KeyedVectors

# Download pretrained w2v GoogleNews-vectors-negative300 and uncompress it:
# https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
w2v_file = 'path/to/GoogleNews-vectors-negative300.bin'
path_to_vocab = '../../data/cnndm/vocab'
save_file = 'path/to/w2v_embedding_matrix.pk'
embed_size = 300 # in main.py change params["embed_size"] according to this embed_size

vocab = Vocab(path_to_vocab, 50000)

def load_w2v_model(w2v_file):
    print("Loading w2v Model")
    w2v_embeddings = KeyedVectors.load_word2vec_format(w2v_file, binary=True)
    return w2v_embeddings

if __name__ == "__main__":

    w2v_embeddings = load_w2v_model(w2v_file)
    embedding_matrix = np.zeros((len(vocab.word2id) , embed_size), dtype=np.float32)

    print("w2v matrices loaded, producing embedding matrix")
            
    for word, i in vocab.word2id.items():
        if word in w2v_embeddings.vocab:
            embedding_matrix[i] = w2v_embeddings.get_vector(word)

    print("Embedding matrices successfully created")

    with open(save_file, "wb") as file:
        pickle.dump(embedding_matrix, file)

    print("Embedding matrices successfully saved using pickle")