import bz2
import os
import pickle
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from tokenizer import LemmaTokenizer
from utils import get_stopwords
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='..')

    parser.add_argument('--input_folder', dest='input_folder', help=' ... ')
    parser.add_argument('--tfidf', dest='tfidf', action='store_true', default=False)
    parser.add_argument('--word_counters', dest='word_counters', help=' ... ')

    args = parser.parse_args()

    input_folder = args.input_folder
    compute_tdf_idf = args.tfidf
    stop = get_stopwords("stopwords.txt")

    if not args.word_counters:
        compute_word_counters = True
        word_counters = []
    else:
        compute_word_counters = False
        word_counters = pickle.load(open(args.word_counters, "rb"))

    print(f"Input folder: {input_folder}\nTFIDF {compute_tdf_idf}\nCompute Word Counters {compute_word_counters}")

    if compute_tdf_idf:
        experiment_folder = input_folder + "/experiment_tf_idf"
    else:
        experiment_folder = input_folder + "/experiment_binary"

    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)

    f = open(f"{input_folder}/annotations.tsv", "r")

    print("Loading and vectorizing text")
    labels = []

    tokenizer = LemmaTokenizer()
    for line in f.readlines():
        s = line.split("\t")
        file_path = s[0]
        print(f"Loading {file_path}")
        domains = s[1][:-2].split("#")

        if compute_word_counters:
            print(f"Reading {file_path}")
            word_counter = Counter([])
            s = b''
            with bz2.open(f"{input_folder}/{file_path}", 'rb') as f:
                while True:
                    s += f.read(1024 * 1024)
                    if len(s) == 0:
                        break
                    L = s.split(b'\n')
                    for li in L[:-1]:  # the 1 MB block that we read might stop in the middle of a line ... (*)
                        word_counter.update(tokenizer(str(li.decode("utf-8"))))
                    s = L[-1]  # (*) ... so we keep the rest for the next iteration
            word_counters.append(dict(word_counter))

        labels.append(domains)

    y = pd.DataFrame()
    y['Class Label'] = labels

    X = pd.DataFrame(word_counters).fillna(0)

    if compute_word_counters:
        print(f"Dumping word_counters in {experiment_folder}/word_counters.p")
        pickle.dump(word_counters, open(f"{experiment_folder}/word_counters.p", "wb"))

    if compute_tdf_idf:
        X = pd.DataFrame(TfidfTransformer().fit_transform(X).fit_transform(X).todense())
    else:
        X = np.sign(X)

    print(f"Dumping X in {experiment_folder}/X_pre.p")
    pickle.dump(X, open(f"{experiment_folder}/X_pre.p", "wb"))

    print(f"Dumping y in {experiment_folder}/y_pre.p")
    pickle.dump(y, open(f"{experiment_folder}/y_pre.p", "wb"))

    print("Dataset created!")
