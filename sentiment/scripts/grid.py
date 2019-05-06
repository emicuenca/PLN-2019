"""Evaulate a Sentiment Analysis model.

Usage:
  grid.py -d <corpus> -t <corpus> -c <clf>
  grid.py -h | --help

Options:
  -d <corpus>   Evaluation corpus.
  -t <corpus>   Trained model file.
  -c <clf>      Classifier to use [default: svm]:
                    maxent: Maximum Entropy
                    svm: Support Vector Machine
  -h --help     Show this screen.
"""

from sklearn.model_selection import ParameterGrid
from sentiment.classifier import SentimentClassifier
from docopt import docopt
from sentiment.evaluator import Evaluator
from sentiment.tass import InterTASSReader
import pandas as pd



def search(clf, params_list):
    results = []
    for params in params_list:
        sc = SentimentClassifier(clf=clf)
        ev = Evaluator()
        sc._pipeline.set_params(**params)
        sc.fit(X_train, y_train)
        y_pred = sc.predict(X_dev)
        ev.evaluate(y_dev, y_pred)
        
        results.append({
            'acc': ev.accuracy(),
            'f1': ev.macro_f1(),
            **params
        })

    results_df = pd.DataFrame(results)
    print(results_df.sort_values(['acc', 'f1'], ascending=False))

def maxent(X_train, y_train, X_dev, y_dev):
    param_grid = {
        'clf__C': [1, 10, 100, 1000, 10000, 20000],
        'clf__penalty': ['l1', 'l2'],
    }
    params_list = list(ParameterGrid(param_grid))
    search('maxent', params_list)

def svm(X_train, y_train, X_dev, y_dev):
    param_grid = [
        {
            'clf__C': [0.25, 0.5, 1, 2, 4, 8, 16, 32],
            'clf__penalty': ['l2'],
            'clf__dual': [True, False],
        },
        {
            'clf__C': [0.25, 1, 4, 64, 256, 512, 1024],
            'clf__dual': [False],
            'clf__penalty': ['l1'],
        }
    ]
    params_list = list(ParameterGrid(param_grid))
    search('svm', params_list)

classifiers = {
    'maxent': maxent,
    'svm': svm
}

if __name__ == '__main__':
    opts = docopt(__doc__)
    train_corpus = opts['-t']
    dev_corpus = opts['-d']
    clf = opts['-c']
    classifier = classifiers[clf]
    
    train_reader = InterTASSReader(train_corpus)
    X_train, y_train = list(train_reader.X()), list(train_reader.y())
    
    dev_reader = InterTASSReader(dev_corpus)
    X_dev, y_dev = list(dev_reader.X()), list(dev_reader.y())
    classifier(X_train, y_train, X_dev, y_dev)