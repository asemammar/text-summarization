import sys, pickle, numpy as np
from batcher import Vocab
import pickle
import tensorflow_hub as hub
import tensorflow_text

use_file = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
path_to_vocab = '../../data/mlsumtr/vocab'
save_file = 'path/to/use-tr_embedding_matrix.pk'
embed_size = 512 # in main.py change params["embed_size"] according to this embed_size

vocab = Vocab(path_to_vocab, 50000)

if __name__ == "__main__":
    print("Loading universal-sentence-encoder-multilingual_3 Model")
    use_embeddings = hub.load(use_file)
    
    print("Model loaded, producing embedding matrix")
    embedding_matrix = np.array(use_embeddings(list(vocab.word2id.keys())))

    print("Embedding matrices successfully created")

    with open(save_file, "wb") as file:
        pickle.dump(embedding_matrix, file)

    print("Embedding matrices successfully saved using pickle")