#!/usr/bin/env python3
# coding:utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to build the tf-idf document matrices for retrieval."""

import numpy as np
import scipy.sparse as sp
import argparse
import os
import math
import logging

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import Counter

from drqa import retriever
from drqa import tokenizers


# ----------------------------------------------------------------------
# TD-IDF refer links:
# http://www.ruanyifeng.com/blog/2013/03/tf-idf.html
#
# N-gram refer links:
# http://blog.csdn.net/baimafujinji/article/details/51281816
# https://www.cnblogs.com/zhangkaikai/p/6669750.html
# https://www.ibm.com/developerworks/cn/opensource/os-cn-pythonwith/
#
# with expression
# http://blog.csdn.net/suwei19870312/article/details/23258495/
# ----------------------------------------------------------------------


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

# ------------------------------------------------------------------------------
# Multiprocessing functions
# ------------------------------------------------------------------------------

DOC2IDX = None
PROCESS_TOK = None
PROCESS_DB = None


def init(tokenizer_class, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


# ------------------------------------------------------------------------------
# Build article --> word count sparse matrix.
# ------------------------------------------------------------------------------


def count(ngram, hash_size, doc_id):
    """Fetch the text of a document and compute hashed ngrams counts."""
    global DOC2IDX
    row, col, data = [], [], []
    # Tokenize: get tokens of text with specify doc_id
    tokens = tokenize(retriever.utils.normalize(fetch_text(doc_id)))

    # Get ngrams from tokens, with stopword/punctuation filtering.
    ngrams = tokens.ngrams(
        n=ngram, uncased=True, filter_fn=retriever.utils.filter_ngram
    )

    # Hash ngrams and count occurences
    ''' Counter函数示例
        example:
        a = ['a', 'b', 'a']
        counts = Counter(a)
        counts == Counter({'a': 2, 'b': 1})
    '''
    counts = Counter([retriever.utils.hash(gram, hash_size) for gram in ngrams])

    # Return in sparse matrix data format.
    row.extend(counts.keys())
    col.extend([DOC2IDX[doc_id]] * len(counts))
    data.extend(counts.values())
    return row, col, data


def get_count_matrix(args, db, db_opts):
    """Form a sparse word to document count matrix (inverted index).

    M[i, j] = # times word i appears in document j.
    """
    # Map doc_ids to indexes
    global DOC2IDX
    db_class = retriever.get_class(db)  # get specify db class instance to get documents
    with db_class(**db_opts) as doc_db:  # context management
        doc_ids = doc_db.get_doc_ids()  # get all doc ids

    '''
        enumerate(list) wrap a list to dic as follow:
        list=['a','b','c']
        enumerate(list)=dict{0: 'a', 1: 'b', 2: 'c'}
        
        so iterate enumerate(list) return two values:index(start from 0) and value of origin list
    '''
    DOC2IDX = {doc_id: i for i, doc_id in enumerate(doc_ids)}  # get doc to index maps from doc_ids

    # Setup worker pool
    tok_class = tokenizers.get_class(args.tokenizer)
    workers = ProcessPool(
        args.num_workers,
        initializer=init,
        initargs=(tok_class, db_class, db_opts)
    )

    # Compute the count matrix in steps (to keep in memory)
    logger.info('Mapping...')
    row, col, data = [], [], []
    step = max(int(len(doc_ids) / 10), 1)  # get count of steps
    batches = [doc_ids[i:i + step] for i in range(0, len(doc_ids), step)]  # calc the batch range of each step

    # redefine function signature. use some defaults args to wrap a function object and return
    # a callable object.
    # refer link:http://www.wklken.me/posts/2013/08/18/python-extra-functools.html
    _count = partial(count, args.ngram, args.hash_size)

    for i, batch in enumerate(batches):
        logger.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        for b_row, b_col, b_data in workers.imap_unordered(_count, batch):
            # three lists extend when each step
            row.extend(b_row)  # list[.../step..../step...] hash(n-gram(token from doc))
            col.extend(b_col)  # list[.../step..../step...] index of doc
            data.extend(b_data)  # list[.../step..../step...] value of count of n-gram(token from doc)
    workers.close()
    workers.join()

    logger.info('Creating sparse matrix...')

    '''生成的稀疏矩阵示例
       hash(N-gram) --------------(col)
       index of doc | element=count(hash) 
                    |
                    |
                    |
                    (row)
    '''
    count_matrix = sp.csr_matrix(
        (data, (row, col)), shape=(args.hash_size, len(doc_ids))
    )

    # 将矩阵中实体元素相同的进行相加合并
    count_matrix.sum_duplicates()
    # 输出矩阵，以及其他
    return count_matrix, (DOC2IDX, doc_ids)


# ------------------------------------------------------------------------------
# Transform count matrix to different forms.
# ------------------------------------------------------------------------------


def get_tfidf_matrix(cnts):
    """Convert the word count matrix into tfidf one.

    tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
    * tf = term frequency in document 某一文档中词的频率
    * N = number of documents 文档的数量
    * Nt = number of occurences of term in all documents 词在所有文档中出现的次数
    """
    Ns = get_doc_freqs(cnts)
    idfs = np.log((cnts.shape[1] - Ns + 0.5) / (Ns + 0.5))
    idfs[idfs < 0] = 0
    idfs = sp.diags(idfs, 0)
    tfs = cnts.log1p()
    tfidfs = idfs.dot(tfs)
    return tfidfs


def get_doc_freqs(cnts):
    """Return word --> # of docs it appears in."""
    # 二值（0,1）矩阵，注意(cnts > 0)是一个逻辑判断表达式
    binary = (cnts > 0).astype(int)
    # sum(1) 矩阵的行向量相加，指定求和方向
    freqs = np.array(binary.sum(1)).squeeze()
    return freqs


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('db_path', type=str, default=None,
                        help='Path to sqlite db holding document texts')
    parser.add_argument('out_dir', type=str, default=None,
                        help='Directory for saving output files')
    parser.add_argument('--ngram', type=int, default=2,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    parser.add_argument('--tokenizer', type=str, default='simple',
                        help=("String option specifying tokenizer type to use "
                              "(e.g. 'corenlp')"))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    logging.info('Counting words...')
    count_matrix, doc_dict = get_count_matrix(
        args, 'sqlite', {'db_path': args.db_path}
    )

    logger.info('Making tfidf vectors...')
    tfidf = get_tfidf_matrix(count_matrix)

    logger.info('Getting word-doc frequencies...')
    freqs = get_doc_freqs(count_matrix)

    basename = os.path.splitext(os.path.basename(args.db_path))[0]
    basename += ('-tfidf-ngram=%d-hash=%d-tokenizer=%s' %
                 (args.ngram, args.hash_size, args.tokenizer))
    filename = os.path.join(args.out_dir, basename)

    logger.info('Saving to %s.npz' % filename)
    metadata = {
        'doc_freqs': freqs,
        'tokenizer': args.tokenizer,
        'hash_size': args.hash_size,
        'ngram': args.ngram,
        'doc_dict': doc_dict
    }
    retriever.utils.save_sparse_csr(filename, tfidf, metadata)
