import re
import simplemma

lang_data = simplemma.load_data('en', 'it', 'de', 'es', 'fr', 'nl', 'ru')

WORD = re.compile(r'[a-zA-Z][a-zA-Z]+')
WORD_int = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9]+')


def regex_tokenize(text):
    words = [simplemma.lemmatize(word, lang_data) for word in WORD.findall(text)]
    return words


class LemmaTokenizer:
    def __call__(self, doc):
        return regex_tokenize(doc)


class SplitTokenizer:
    def __call__(self, doc):
        return WORD.findall(doc)

