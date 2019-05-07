"""Show most relevant features in a maxent model

Usage:
  features.py -m <file>
  features.py -h | --help

Options:
  -m <file>     Trained model file.
  -h --help     Show this screen.
"""

from docopt import docopt
import pickle
import pandas as pd
from sentiment.analysis import print_maxent_features, maxent_features_to_html


def main():
    # Load model from file
    opts = docopt(__doc__)
    filename = opts['-m']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # Get vectorizer and classifier
    pipeline = model._pipeline
    vect = pipeline.named_steps['vect']
    clf = pipeline.named_steps['clf']
    
    # Print 10 most relevant features
    # print_maxent_features(vect, clf, n=10)
    html = maxent_features_to_html(vect, clf, n=10)
    filename = 'features.html'
    with open(filename, 'w+') as f:
      f.write(html)

if __name__ == '__main__':
    main()
