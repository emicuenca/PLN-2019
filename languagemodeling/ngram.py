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
        log_prob = 0
        for sent in sents:
            sent = sent + ['</s>']
            log_prob += self.sent_log_prob(sent)

        return log_prob

    def cross_entropy(self, sents):
        """Cross-entropy of a list of sentences.

        sents -- the sentences.
        """
        # WORK HERE!!
        m = len(sents)
        for sent in sents:
            m += len(sent)
        cross_entropy = - self.log_prob(sents) / m

        return cross_entropy

    def perplexity(self, sents):
        """Perplexity of a list of sentences.

        sents -- the sentences.
        """
        # WORK HERE!!

        cross_entropy = self.cross_entropy(sents)
        perplexity = 2 ** cross_entropy

        return perplexity


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
            for i in range(1, n):
                ngram = tuple(['<s>'] * (n - i) + sent[:i])
                nminusonegram = ngram[:n - 1]
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


class SmoothedNGram(LanguageModel):
    @staticmethod
    def ngramCount(sents, n):
        print('Computing counts...')

        count = defaultdict(int)
        for sent in sents:
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

        return dict(count)
    
    @staticmethod
    def vocabularySize(sents):
        print('Computing vocabulary...')

        voc = set()
        voc.add('</s>')
        for sent in sents:
            for word in sent:
                voc.add(word)

        V = len(voc)

        return V, voc
    
    @staticmethod
    def splitData(sents, trainPercentage):
        m = int(trainPercentage * len(sents))
        train_sents = sents[:m]
        held_out_sents = sents[m:]
        
        return train_sents, held_out_sents


class AddOneNGram(NGram, SmoothedNGram):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        # call superclass to compute counts
        super().__init__(n, sents)

        # compute vocabulary
        self._V, self._voc = self.vocabularySize(sents)

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


class InterpolatedNGram(NGram, SmoothedNGram):

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
            train_sents, held_out_sents = self.splitData(sents, .9)

        # WORK HERE!!
        # COMPUTE COUNTS FOR ALL K-GRAMS WITH K <= N
                    
        self._count = self.ngramCount(train_sents, n)

        # compute vocabulary size for add-one in the last step
        self._addone = addone
        if addone:
            self._V, self._voc = self.vocabularySize(train_sents)

        # compute gamma if not given
        if gamma is not None:
            self._gamma = gamma
        else:
            print('Computing gamma...')
            # use grid search to choose gamma
            values = [1, 10, 100, 1000, 10000]
            best = (-math.inf, values[0])
            for value in values:
                self._gamma = value
                log_prob = self.log_prob(held_out_sents)
                print(">>", value, log_prob)
                best = max(best, (log_prob, value))
            self._gamma = best[1]
            print("Gamma:", self._gamma)

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


class BackOffNGram(NGram, SmoothedNGram):
    def __init__(self, n, sents, beta=None, addone=True):
        """
        Back-off NGram model with discounting as described by Michael Collins.
 
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        beta -- discounting hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        self._n = n
        
        if beta is not None:
            # everything is training data
            train_sents = sents
        else:
            # 90% training, 10% held-out
            train_sents, held_out_sents = self.splitData(sents, .9)
        
        # compute kgram count for 0 <= k <= n
        self._count = self.ngramCount(train_sents, n)

        # compute vocabulary size for add-one in the last step
        self._addone = addone
        if addone:
            self._V, self._voc = self.vocabularySize(train_sents)
        
        # precompute A
        A = defaultdict(set)
        for kgram in self._count:
            if not kgram == ():
                kminusonegram = kgram[:-1]
                A[kminusonegram].add(kgram[-1])
        self._A = dict(A)
        
        # compute beta if not given
        if beta is None:
            print('Computing beta...')
            values = [0.1, 0.2, 0.3, 0.5, 0.7]
            best = (-math.inf, values[0])
            for value in values:
                self._beta = value
                log_prob = self.log_prob(held_out_sents)
                print(">>", value, log_prob)
                best = max(best, (log_prob, value))
            self._beta = best[1]
            print("Beta:", self._beta)
        else:
            self._beta = beta

    def A(self, tokens):
        """Set of words with counts > 0 for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """

        return self._A.get(tokens, set())

    def alpha(self, tokens):
        """Missing probability mass for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """
        A = self.A(tokens)
        beta = self._beta
        
        if 0 < len(A):
            alpha = beta * len(A) / self.count(tokens)
        else:
            alpha = 1
        
        return alpha

    def denom(self, tokens):
        """Normalization factor for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """
        accum = 0
        A = self.A(tokens)
        beta = self._beta
        tail = tokens[1:]
        for x in A:
            accum += self.count(tail + (x,)) - beta
        if 0 < len(A):
            accum /= self.count(tail)
        denom = 1 - accum
            
        return denom

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        if prev_tokens is None:
            prev_tokens = ()
        
        beta = self._beta
        prob = 1
        A = self.A(prev_tokens)

        while 0 < len(prev_tokens) and not token in A:
            prob *= self.alpha(prev_tokens) / self.denom(prev_tokens)
            prev_tokens = prev_tokens[1:]
            A = self.A(prev_tokens)

        if self._addone and len(prev_tokens) == 0:
            prob *= self.count((token,)) + 1
            prob /= self.count(prev_tokens) + self._V
        else:
            if len(prev_tokens) == 0:
                prob *= self.count((token,))
            else:
                prob *= self.count(prev_tokens + (token,)) - beta
            prob /= self.count(prev_tokens)

        return prob