"""Train an n-gram model.

Usage:
  train.py [-m <model>] -n <n> -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -m <model>    Model to use [default: ngram]:
                  ngram: Unsmoothed n-grams.
                  addone: N-grams with add-one smoothing.
                  inter: N-grams with interpolation smoothing.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle

from nltk.corpus import gutenberg

from nltk.corpus.reader import PlaintextCorpusReader
from nltk.data import find
from nltk import RegexpTokenizer
import os

from languagemodeling.ngram import NGram, AddOneNGram, InterpolatedNGram, BackOffNGram


models = {
    'ngram': NGram,
    'addone': AddOneNGram,
    'inter': InterpolatedNGram,
    'backoff': BackOffNGram
}


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    # TODO: Corpus must be larger than 5MB
    corpora_dir = find(os.path.join(os.getcwd(), 'corpora'))
    custom_tokenizer = RegexpTokenizer('[^.!?]+')
    reader = PlaintextCorpusReader(corpora_dir, '.*\.txt', sent_tokenizer=custom_tokenizer)
    sents = reader.sents('corpus-utf8.txt')

    # train the model
    n = int(opts['-n'])
    model_class = models[opts['-m']]
    model = model_class(n, sents)

    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
