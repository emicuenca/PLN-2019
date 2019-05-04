from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import re

classifiers = {
    'maxent': LogisticRegression,
    'mnb': MultinomialNB,
    'svm': LinearSVC,
}

negations = ['no', 'ni', 'nunca', 'jamas', 'pero', 'tampoco']

def customPreprocess(s):
    vowelRegex = "([aeiou])\\1{2,}"
    urlRegex = "https?://[^\\s\\<]+"
    mentionRegex = "(?:(?<=\\s|\\>)|(?<=^))@[a-zA-Z0-9_\\-]{,15}\\s?"

    # Default preprocessor converts all characters to lowercase
    prepro = s.lower()
    prepro = re.sub(vowelRegex, "\\1", prepro)
    prepro = re.sub(urlRegex, " ", prepro)
    prepro = re.sub(mentionRegex, " ", prepro)
    return prepro

class NegationTokenizer(object):
    def __init__(self):
        self.remaining = 0
        self.words_after_negation = 3
        self.negations = negations
    
    def __call__(self, tokens):
        for token in tokens:
            if 0 < self.remaining:
                if len(token) == 1 or token in self.negations:
                    self.remaining = 0
                else:
                    token =  "NOT_" + token
                    self.remaining -= 1
            elif token in self.negations:
                self.remaining = self.words_after_negation
            yield token

class CustomVectorizer(CountVectorizer):
    def build_tokenizer(self):
        tokenize = super(CustomVectorizer, self).build_tokenizer()
        negationTokenizer = NegationTokenizer()
        return lambda doc: list(negationTokenizer(tokenize(doc)))

class SentimentClassifier(object):

    def __init__(self, clf='svm'):
        """
        clf -- classifying model, one of 'svm', 'maxent', 'mnb' (default: 'svm').
        """
        self._clf = clf
        stop_words = list(set(stopwords.words('spanish')) - set(negations))
        self._pipeline = pipeline = Pipeline([

            ('vect', CustomVectorizer(binary=True, preprocessor=customPreprocess, stop_words=stop_words)),
            #('vect', CountVectorizer(stop_words=stopwords.words('spanish'))),
            #('vect', CountVectorizer(preprocessor=customPreprocess)),
            #('vect', CustomVectorizer()),
            #('vect', CountVectorizer(binary=True)),
            #('vect', CountVectorizer()),
            ('clf', classifiers[clf]()),
        ])

    def fit(self, X, y):
        self._pipeline.fit(X, y)

    def predict(self, X):
        return self._pipeline.predict(X)
    
    def predict_proba(self, X):
        return self._pipeline.predict_proba(X)
