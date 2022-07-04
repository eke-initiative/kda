import os
import pickle
import pycountry
from stop_words import get_stop_words, AVAILABLE_LANGUAGES
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_stopwords(stopwords_file):

    if os.path.exists(stopwords_file + ".p"):
        return pickle.load(open(stopwords_file + ".p", "rb"))

    # Create a stopword list
    stop = set()
    with open(stopwords_file, "r") as fp:
        line = fp.readline()
        stop.add(line[:-1])
        while line:
            line = fp.readline()
            stop.add(line[:-1])

    for c in pycountry.countries:
        stop.add(c.alpha_2.lower())
        stop.add(c.alpha_3.lower())

    # # Importing stopwords for available languages https://github.com/Alir3z4/python-stop-words
    for language in AVAILABLE_LANGUAGES:
        for sw in get_stop_words(language):
            stop.add(sw)

    words_to_exclude = ["property", "label", "comment", "class", "restriction", "ontology", "nil", "individual",
                        "value", "domain", "range", "first", "rest", "resource", "datatype", "integer", "equivalent",
                        "title", "thing", "creator", "disjoint", "predicate", "dublin", "taxonomy", "axiom", "foaf",
                        "dc", "uri", "void", "dataset", "subject", "term", "agent",
                        "boolean", "xml", "httpd", "https", "sub"]

    for w in words_to_exclude:
        stop.add(w)

    logger.info(f"Number of Stopwords {len(stop)}")

    pickle.dump([w for w in stop], open(stopwords_file + ".p", "wb"))
    return stop
