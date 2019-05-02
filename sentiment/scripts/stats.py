"""Show corpus statistics

Usage:
  stats.py [options] -f <folder>
  stats.py -h | --help

Options:
  -f <folder>   Folder containing TASS corpora.
  -h --help     Show this screen.
"""
from docopt import docopt
from os import path
from sentiment.tass import InterTASSReader
from collections import Counter

if __name__ == '__main__':
    opts = docopt(__doc__)

    # load corpora
    filenames = [f'intertass-{location}-train-tagged.xml' for location in ['ES', 'CR', 'PE']]
    folder = opts['-f']

    for filename in filenames:
        filepath = path.join(folder, filename)
        reader = InterTASSReader(filepath)
        y = list(reader.y())
        print('***')
        print(filename)
        print('tweets:', len(y))
        print(Counter(y))
