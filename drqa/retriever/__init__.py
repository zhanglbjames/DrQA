#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from .. import DATA_DIR

# set default arguments
DEFAULTS = {
    'db_path': os.path.join(DATA_DIR, 'wikipedia/docs.db'),
    'tfidf_path': os.path.join(
        DATA_DIR,
        'wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz'
    ),
}

# follow functions can be invoke as:
# retriever.setdefault(k,v)
def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


def get_class(name):
    if name == 'tfidf':
        return TfidfDocRanker
    if name == 'sqlite':
        return DocDB
    raise RuntimeError('Invalid retriever class: %s' % name)

# __info__.py helps
# refer links: http://blog.csdn.net/djangor/article/details/39673659
#              http://www.cnpythoner.com/post/2.html

# simplify package import
from .doc_db import DocDB
from .tfidf_doc_ranker import TfidfDocRanker
# suit to py2.7
from .utils import *
