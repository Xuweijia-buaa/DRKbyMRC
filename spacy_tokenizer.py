#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Tokenizer that is backed by spaCy (spacy.io).

Requires spaCy package and the spaCy english model.
"""

import spacy
import copy
from tokenizer import Tokens, Tokenizer

# a class which have tokenize method
# use tokenize method,  return a Tokens object metarilized by a text
#  Tokenizer: a class have tokenize method
#  SpacyTokenizer: a class rewrite tokenize method
class SpacyTokenizer(Tokenizer):

    def __init__(self, **kwargs):
        #  **kwargs :  any key-based parameters
        # have key ['model'], ['annotators']
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: spaCy model to use (either path, or keyword like 'en').
        """
        model = kwargs.get('model', 'en')                                  # default mpdel='en'
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))   # set:('pos,'ner','lemma')
        nlp_kwargs = {'parser': False}                                     # no parser
        if not {'lemma', 'pos', 'ner'} & self.annotators:                  # pos/ner/lemma,   use tagger
            nlp_kwargs['tagger'] = False
        if not {'ner'} & self.annotators:                                  #                  use entity 
            nlp_kwargs['entity'] = False
            
        # default: nlp=spacy.load('en')
        # nlp_kwargs: {'parser'=False,'tagger'=True,'entity'=True}
        self.nlp = spacy.load(model, **nlp_kwargs)

    def tokenize(self, text):
        # We don't treat new lines as tokens.
        clean_text = text.replace('\n', ' ')
        # tokens
        tokens = self.nlp.tokenizer(clean_text)
        
        if {'lemma', 'pos', 'ner'} & self.annotators:
            self.nlp.tagger(tokens)
        if {'ner'} & self.annotators:
            self.nlp.entity(tokens)

        data = []   # each element is a token, include all this token's property
        for i in range(len(tokens)):
            # Get whitespace
            # Obama is a New York City citizen, who loves me. And I love him so much
            start_ws = tokens[i].idx                 #  i th token's start character pos in orig sens.  0,6,9,...66(m)
            if i + 1 < len(tokens):
                end_ws = tokens[i + 1].idx           #  i+1 th token's start character pos in orig sens 6,9,11,... 66+4(h)
            else:
                end_ws = tokens[i].idx + len(tokens[i].text)
                
            # each token's tuple
            data.append((
                tokens[i].text,                                          # token本身                              Obama/is/.../much
                text[start_ws: end_ws],                                  # token对应文本(with whitespace)         [Obama ]/[is ]/.../[so ]/[much]
                (tokens[i].idx, tokens[i].idx + len(tokens[i].text)),    # token的span(no whitespace),char级别    (0,5)/(6,8),(9,10)/.../
                tokens[i].tag_,                                          # token的pos                             NNP/VBZ/DT/NNP/NNP/...
                tokens[i].lemma_,                                        # 原型                                  'obama'/'is'/
                tokens[i].ent_type_,                                     # token的 NER类型                       'ORG'/''/''/'GPE'/'GPE'
            ))

        # Set special option for non-entity tag: '' vs 'O' in spaCy
        # return a Token object
        return Tokens(data, self.annotators, opts={'non_ent': ''})
