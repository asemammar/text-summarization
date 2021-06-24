import sys, pickle, numpy as np
from batcher import Vocab
import pickle
import tensorflow_hub as hub

nnlm_file = 'https://tfhub.dev/google/nnlm-en-dim128/2'
path_to_vocab = '../../data/cnndm/vocab'
save_file = 'path/to/nnlm_embedding_matrix.pk'
embed_size = 128 # in main.py change params["embed_size"] according to this embed_size

vocab = Vocab(path_to_vocab, 50000)

if __name__ == "__main__":
    print("Loading nnlm-en-dim128_2 Model")
    nnlm_embeddings = hub.load(nnlm_file)
    
    print("Model loaded, producing embedding matrix")
    embedding_matrix = np.array(nnlm_embeddings(list(vocab.word2id.keys())))

    print("Embedding matrices successfully created")

    with open(save_file, "wb") as file:
        pickle.dump(embedding_matrix, file)

    print("Embedding matrices successfully saved using pickle")