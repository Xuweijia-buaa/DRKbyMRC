#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Base tokenizer/tokens classes and utilities."""
# example: 'Obama is a New York City citizen, who loves me. And I love him so much'
import copy
class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5
    # data, each element is a token, include all it'e property
    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}
    # all tokens number
    def __len__(self):
        """The number of tokens."""
        return len(self.data)
    
    # return a Tokens object, just include [i,j) tokens  
    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens
    
    # 恢复出original text
    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    # list, all tokens text本身: [Obama,is,....much]
    def words(self, uncased=False):
        """Returns a list of the text of each token
        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]
        
    # 每个token 起始和结束位置(char 级别) : [(0,5),(6,8),(9,10),...]
    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    #  每个token的pos: [NNP,VBZ,DT,NNP,NNP...]  or None
    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]
    
    # 每个token的lemmas: ['obama','is',...]  or None
    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]
    # 每个token的entity:['ORG','','','GPE','GPE'] or None
    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    # return all 1-n n-gram tokens: [‘I’，'I love','I love you', 'love','love you','love you and',...]
    # if filter_fn=(utils.filter_ngram(gram, mode='any')), skip some grams , accordding to whether contain stop/biaodian tokens
    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.
        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """
        # skip some grams , accordding to whether contain stop/biaodian tokens
        def _skip(gram):
            if not filter_fn:       #  keep all grrams
                return False
            return filter_fn(gram)  # skip some grams , accordding to whether contain stop/biaodian tokens. skip,return True

        words = self.words(uncased) # token list
        # all token's all 1-n grams index:
        # len words=5, n=3:  [(0, 1),(0, 2),(0, 3),(1, 2),(1, 3),(1, 4),(2, 3), (2, 4),(2, 5),(3, 4),(3, 5),(4, 5)]
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]                  # _skip: 输入ngram tokens  
        # Concatenate into strings
        if as_strings:
            # all 1-n n-gram tokens: ‘I’，'I love','I love you', 'love','love you','love you and'
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]
        return ngrams


    #  return each entity's original text [（"mao ze dong"，ORG),（"jing gang shan"，ORG）]
    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        #  entities: each token's entity:['ORG','','','GPE','GPE'] or None
        entities = self.entities()
        if not entities:
            return None       
        non_ent = self.opts.get('non_ent', 'O') # non_entity str,default 'O'. spacy:''
        groups = []
        idx = 0   # token 位置
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                # group： text中所有entity，按顺序加入list
                # self.slice(start, idx).untokenize(): each entity's original text
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """
    # 什么都不做
    #父类 其中的tokenize会被corenlp等类重写，输入是text
    def tokenize(self, text):
        raise NotImplementedError

    # 释放时pass：不做任何事
    def shutdown(self):
        pass
    # 析构函数
    def __del__(self):
        self.shutdown()
