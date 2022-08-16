import bz2
import os
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

from tokenizer import LemmaTokenizer
from utils import get_stopwords

input_folder = sys.argv[1]

compute_tdf_idf = False
if len(sys.argv) > 2 and sys.argv[2] == "tfidf":
    compute_tdf_idf = True

if compute_tdf_idf:
    experiment_folder = input_folder + "/experiment_tf_idf"
else:
    experiment_folder = input_folder + "/experiment_binary"

stop = get_stopwords("stopwords.txt")

if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)

f = open(f"{input_folder}/annotations.tsv", "r")

print("Loading and vectorizing text")
labels = []
word_counters = []
tokenizer = LemmaTokenizer()
for line in f.readlines():
    s = line.split("\t")
    file_path = s[0]
    print(f"Loading {file_path}")
    domains = s[1][:-2].split("#")

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

    # word_counter = Counter([])
    # for li in bz2.open(f"{input_folder}/{file_path}", "r"):
    #     word_counter.update(tokenizer(str(li.decode("utf-8")).strip("\n")))

    labels.append(domains)
    word_counters.append(dict(word_counter))

X = pd.DataFrame(word_counters).fillna(0)

print(f"Dumping word_counters in {experiment_folder}/word_counters.p")
pickle.dump(word_counters, open(f"{experiment_folder}/word_counters.p", "wb"))

if compute_tdf_idf:
    X = pd.DataFrame(TfidfTransformer().fit_transform(X).fit_transform(X).todense())
else:
    X = np.sign(X)

y = pd.DataFrame(labels, columns=['Class Label'])

print(f"Dumping X in {experiment_folder}/X_pre.p")
pickle.dump(X, open(f"{experiment_folder}/X_pre.p", "wb"))

print(f"Dumping y in {experiment_folder}/y_pre.p")
pickle.dump(y, open(f"{experiment_folder}/y_pre.p", "wb"))

print("Dataset created!")
