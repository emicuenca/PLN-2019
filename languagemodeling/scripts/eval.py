"""Evaulate a language model using a test set.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import math

from nltk.corpus import gutenberg
from nltk.corpus.reader import PlaintextCorpusReader
from nltk.data import find
from nltk import RegexpTokenizer
import os

if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # load the data
    # WORK HERE!! LOAD YOUR EVALUATION CORPUS
    #sents = gutenberg.sents('austen-persuasion.txt')
    corpora_dir = find(os.path.join(os.getcwd(), 'corpora'))
    custom_tokenizer = RegexpTokenizer('[^.!?]+')
    reader = PlaintextCorpusReader(corpora_dir, '.*\.txt', sent_tokenizer=custom_tokenizer)
    sents = reader.sents('test-utf8.txt')
    
    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # compute the cross entropy
    # WORK HERE!!
    log_prob = model.log_prob(sents)
    e = model.cross_entropy(sents)
    p = model.perplexity(sents)

    print('Log probability: {}'.format(log_prob))
    print('Cross entropy: {}'.format(e))
    print('Perplexity: {}'.format(p))
