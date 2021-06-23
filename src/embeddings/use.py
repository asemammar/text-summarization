import sys, pickle, numpy as np
from batcher import Vocab
import pickle
import tensorflow_hub as hub
import tensorflow_text

# https://tfhub.dev/google/universal-sentence-encoder/4
use_file = 'D:/Boun/PhD/Season2(S21)/CmpE58T/Project/Data/universal-sentence-encoder_4'
path_to_vocab = 'D:/Boun/PhD/Season2(S21)/CmpE58T/Project/Data/tfrecords_finished_files/vocab'
save_file = 'D:/Boun/PhD/Season2(S21)/CmpE58T/Project/Data/embeddings/use_embedding_matrix.pk'
embed_size = 512 # params["embed_size"]

vocab = Vocab(path_to_vocab, 50000)

if __name__ == "__main__":
    print("Loading universal-sentence-encoder_4 Model")
    use_embeddings = hub.load(use_file)
    
    print("Model loaded, producing embedding matrix")
    embedding_matrix = np.array(use_embeddings(list(vocab.word2id.keys())))

    print("Embedding matrices successfully created")

    with open(save_file, "wb") as file:
        pickle.dump(embedding_matrix, file)

    print("Embedding matrices successfully saved using pickle")