"""Show most relevant errors in classification

Usage:
  errors.py -m <file> -d <corpus>
  errors.py -h | --help

Options:
  -d <corpus>   Path to Development corpus.
  -m <file>     Trained model file.
  -h --help     Show this screen.
"""

from sentiment.tass import InterTASSReader
from docopt import docopt
import pickle
import pandas as pd
import numpy as np

def main():
    # Load model from file
    opts = docopt(__doc__)
    filename = opts['-m']
    with open(filename, 'rb') as f:
        model = pickle.load(f)

    # Load development corpus
    corpus = opts['-d']
    reader = InterTASSReader(corpus)
    X_dev, y_dev = list(reader.X()), list(reader.y())

    y_pred = model.predict(X_dev)
    y_prob = model.predict_proba(X_dev)

    # ys:   probability for each class
    # y_p:  predicted value
    # y_t:  true value
    errors = []
    for x, y_t, y_p, ys in zip(X_dev, y_dev, y_pred, y_prob):
        if y_p != y_t:
            classes = model._pipeline.classes_
            error = {
                'item': x,
                'true': y_t,
                'predicted': y_p,
                'diff': ys[np.where(classes == y_p)] - ys[np.where(classes == y_t)]
            }
            for i in range(len(classes)):
                class_label = classes[i]
                prob_for_class = ys[i]
                error[class_label] = prob_for_class
            errors.append(error)
    
    errors_df = pd.DataFrame(errors)
    errors_df.sort_values('diff', ascending=False, inplace=True)
    print(errors_df[:10])

    

if __name__ == '__main__':
    main()