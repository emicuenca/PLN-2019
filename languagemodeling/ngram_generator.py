from collections import defaultdict
import random


class NGramGenerator(object):

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self._n = model._n

        # compute the probabilities
        probs = defaultdict(dict)
        
        # WORK HERE!!
        n = self._n
        ngrams = list(model._count.keys())
        
        for ngram in ngrams:
            if len(ngram) == n:
                token = ngram[n - 1]
                prev_tokens = ngram[:n - 1]
                probs[prev_tokens][token] = model.cond_prob(token, prev_tokens)

        self._probs = dict(probs)

        # sort in descending order for efficient sampling
        self._sorted_probs = sorted_probs = {}

        # WORK HERE!!
        for prev_tokens, cond_prob in probs.items():
            items = list(cond_prob.items())
            items.sort()
            self._sorted_probs[prev_tokens] = items

    def generate_sent(self):
        """Randomly generate a sentence."""
        # WORK HERE!!
        n = self._n
        gen_token = '<s>'
        token_string = [gen_token] * (n - 1)
        
        while not gen_token == '</s>':
            prev_tokens = tuple(token_string[len(token_string) - n + 1:])
            gen_token = self.generate_token(prev_tokens)
            token_string.append(gen_token)

        return token_string[n-1:-1]

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        # WORK HERE!!
        if prev_tokens is None:
            if self._n == 1:
                prev_tokens = ()
            #else:
                #throw an exception
        
        r = random.random()
        probs = list(self._probs[prev_tokens].items())
        accum = probs[0][1]
        i = 0
        while accum < r:
            i += 1
            accum += probs[i][1]
        gen_token = probs[i][0]
        
        return gen_token
        