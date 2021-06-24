import sys, pickle, numpy as np
from batcher import Vocab
import pickle

# Download your preferred pretrained Glove Embedding. We used glove.6B.100d.
# https://nlp.stanford.edu/projects/glove/
glove_file = 'path/to/glove.6B.100d.txt'
path_to_vocab = '../../data/cnndm/vocab'
save_file = 'path/to/glove_embedding_matrix.pk'
embed_size = 100 # in main.py change params["embed_size"] according to this embed_size

vocab = Vocab(path_to_vocab, 50000)

def load_glove_model(glove_file):
    print("Loading Glove Model")
    with open(glove_file,'r',encoding='utf8') as f:
        embeddings_index = {}
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        print("Done.",len(embeddings_index)," words loaded!")
        return embeddings_index

if __name__ == "__main__":

    glove_embeddings = load_glove_model(glove_file)
    embedding_matrix = np.zeros((len(vocab.word2id) , embed_size), dtype=np.float32)

    print("Glove matrices loaded, producing embedding matrix")
            
    for word, i in vocab.word2id.items():
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print("Embedding matrices successfully created")

    with open(save_file, "wb") as file:
        pickle.dump(embedding_matrix, file)

    print("Embedding matrices successfully saved using pickle")