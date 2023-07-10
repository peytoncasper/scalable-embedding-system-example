from sentence_transformers import SentenceTransformer
import time
import numpy as np
import pandas as pd

if __name__ == '__main__':
    model = SentenceTransformer('all-MiniLM-L6-v2')

    df = pd.read_csv('sentiment-emotion-labelled_Dell_tweets.csv', sep=',', header=None)
    data = df[3]

    start_time = time.time()
    sentence_embeddings = model.encode(data, batch_size=1)
    print("--- Batch Size: 1 ---")
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    sentence_embeddings = model.encode(data, batch_size=32)
    print("--- Batch Size: 32 ---")
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    sentence_embeddings = model.encode(data, batch_size=128)
    print("--- Batch Size: 128 ---")
    print("--- %s seconds ---" % (time.time() - start_time))



    #
    #
    #
    # # Our sentences we like to encode
    # sentences = ['This framework generates embeddings for each input sentence',
    #              'Sentences are passed as a list of string.',
    #              'The quick brown fox jumps over the lazy dog.']
    #
    # # Sentences are encoded by calling model.encode()
    #



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
