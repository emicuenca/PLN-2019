# https://docs.python.org/3/library/collections.html
from collections import defaultdict
import math


class LanguageModel(object):

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        return 0.0

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        return -math.inf

    def log_prob(self, sents):
        """Log-probability of a list of sentences.

        sents -- the sentences.
        """
        # WORK HERE!!

    def cross_entropy(self, sents):
        """Cross-entropy of a list of sentences.

        sents -- the sentences.
        """
        # WORK HERE!!

    def perplexity(self, sents):
        """Perplexity of a list of sentences.

        sents -- the sentences.
        """
        # WORK HERE!!


class NGram(LanguageModel):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self._n = n

        count = defaultdict(int)

        for sent in sents:
            sent = sent + ['</s>']
            #if 1 < n:
            #    sent = ['<s>'] + sent
            for i in range(1, n):
                ngram = tuple(['<s>'] * (n - i) + sent[:i])
                nminusonegram = ngram[:n - 1]
                print('A ver...', n, i, ngram, nminusonegram)
                count[ngram] += 1
                count[nminusonegram] += 1
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i:i+n])
                nminusonegram = tuple(sent[i:i+n-1])
                count[ngram] += 1
                count[nminusonegram] += 1
            
        self._count = dict(count)
        

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self._count.get(tokens, 0)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        # WORK HERE!!
        if prev_tokens is None:
            w_n = (token,)
            w_nminusone = ()
        else:
            w_n = prev_tokens + (token,)
            w_nminusone = prev_tokens
        return self.count(w_n) / self.count(w_nminusone)

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        # WORK HERE !!
        n = self._n
        cond_prob = 1.
        
        sent = sent + ['</s>']
        if 1 < n:
            sent = ['<s>'] + sent
        
        for i in range(len(sent) - n + 1):
            token = sent[i + n - 1]
            prev_tokens = tuple(sent[i:i+n-1])
            cond_prob *= self.cond_prob(token, prev_tokens)
            if cond_prob == 0.:
                break

        return cond_prob
            

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        # WORK HERE!!
        n = self._n
        log_prob = 0
        
        sent = sent + ['</s>']
        if 1 < n:
            sent = ['<s>'] + sent
        
        for i in range(len(sent) - n + 1):
            token = sent[i + n - 1]
            prev_tokens = tuple(sent[i:i+n-1])
            cond_prob = self.cond_prob(token, prev_tokens)
            if cond_prob == 0.:
                log_prob = - math.inf
                break
            else:
                log_prob += math.log2(cond_prob)
        
        return log_prob


class AddOneNGram(NGram):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        # call superclass to compute counts
        super().__init__(n, sents)

        # compute vocabulary
        self._voc = voc = set()
        # WORK HERE!!
        voc.add('</s>')
        for sent in sents:
            for word in sent:
                voc.add(word)

        self._V = len(voc)  # vocabulary size

    def V(self):
        """Size of the vocabulary.
        """
        return self._V

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        # WORK HERE!!
        if prev_tokens is None:
            w_n = (token,)
            w_nminusone = ()
        else:
            w_n = prev_tokens + (token,)
            w_nminusone = prev_tokens
        
        return (self.count(w_n) + 1) / (self.count(w_nminusone) + self.V())


class InterpolatedNGram(NGram):

    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        assert n > 0
        self._n = n

        if gamma is not None:
            # everything is training data
            train_sents = sents
        else:
            # 90% training, 10% held-out
            m = int(0.9 * len(sents))
            train_sents = sents[:m]
            held_out_sents = sents[m:]

        print('Computing counts...')
        # WORK HERE!!
        # COMPUTE COUNTS FOR ALL K-GRAMS WITH K <= N
        count = defaultdict(int)
        
        for sent in train_sents:
            sent = sent + ['</s>']
            count[()] += len(sent)
            for k in range(1, n + 1):
                # kgrams with at least one <s> symbol
                for j in range(k):
                    kgram = tuple(['<s>']* (k-j) + sent[:j])
                    count[kgram] += 1
                for i in range(len(sent) - k + 1):
                    kgram = tuple(sent[i:i+k])
                    count[kgram] += 1
                    
        self._count = dict(count)
                    
            

        # compute vocabulary size for add-one in the last step
        self._addone = addone
        if addone:
            print('Computing vocabulary...')
            self._voc = voc = set()
            # WORK HERE!!
            for sent in train_sents:
                for word in sent:
                    voc.add(word)

            self._V = len(voc) + 1

        # compute gamma if not given
        if gamma is not None:
            self._gamma = gamma
        else:
            print('Computing gamma...')
            # WORK HERE!!
            # use grid search to choose gamma
            # TODO
            self._gamma = 1.

    def count(self, tokens):
        """Count for an k-gram for k <= n.

        tokens -- the k-gram tuple.
        """
        # WORK HERE!! (JUST A RETURN STATEMENT)
        return self._count.get(tokens, 0)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        # WORK HERE!!
        gamma = self._gamma
        n = self._n
        
        if prev_tokens is None and n == 1:
            prev_tokens = ()
        
        ngram = prev_tokens + (token,)
        
        prev_lambdas_factor = 1.
        prob = 0
        for i in range(len(ngram)):
            
            kgram = ngram[i:]
            kminusonegram = kgram[:-1] 
            k_count = self.count(kgram)
            kminusone_count = self.count(kminusonegram)
            
            # For every k-gram with 1 < k
            if 1 < len(kgram):
                prob += prev_lambdas_factor * k_count / (kminusone_count + gamma)
                curr_lamb = prev_lambdas_factor * kminusone_count / (kminusone_count + gamma)
                prev_lambdas_factor -= curr_lamb
            # For 1-grams with addone smoothing
            elif len(kgram) == 1 and self._addone:
                prob += prev_lambdas_factor * (k_count + 1) / (kminusone_count + self._V)
            # For 1-grams without addone
            else:
                prob += prev_lambdas_factor * k_count / kminusone_count
        
        return prob



