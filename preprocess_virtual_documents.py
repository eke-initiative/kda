import sys
import bz2
import pandas as pd
from utils import get_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from tokenizer import LemmaTokenizer
import pickle
import os

input_folder = sys.argv[1]
experiment_folder = input_folder + "/experiment"

if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)

f = open(f"{input_folder}/annotations.tsv", "r")

data = []

print("Loading text")

for line in f.readlines():
    s = line.split("\t")
    file_path = s[0]
    print(f"Loading {file_path}")
    domains = s[1][:-2].split("#")
    txt = " ".join([str(line.decode("utf-8")).strip("\n") for line in bz2.open(f"{input_folder}/{file_path}", "r")])
    data.append([domains, txt])


df = pd.DataFrame(data, columns=['Class Label', 'Text'])

X = df['Text']
y = df['Class Label']

stop = get_stopwords("stopwords.txt")

cv = CountVectorizer(lowercase=True, stop_words=stop, tokenizer=LemmaTokenizer(), binary=True)

# Preprocessing
print("Vectorization")
X = pd.DataFrame(cv.fit_transform(X).todense())

print(f"Dumping X in {experiment_folder}/X_pre.p")
pickle.dump(X, open(f"{experiment_folder}/X_pre.p", "wb"))

print(f"Dumping y in {experiment_folder}/y_pre.p")
pickle.dump(y, open(f"{experiment_folder}/y_pre.p", "wb"))

print(f"Dumping vectorizer in {experiment_folder}/cv.p")
pickle.dump(cv, open(f"{experiment_folder}/cv.p", "wb"))

print("Dataset created!")
