import argparse
import logging
import re
from math import ceil
from pathlib import Path
from time import time

import numpy as np
from sentence_transformers import SentenceTransformer, util

from lexrank import STOPWORDS, LexRank
from lexrank.algorithms.power_method import stationary_distribution

logger = logging.getLogger('smartedu-summarizer')


def summarize_idf(args):
    logger.info(f'Loading file \'{args.file}\'')
    sentences = list()
    with args.file.open('rU', encoding='utf-8') as file_:
        for line in file_:
            if line.rstrip():
                sentences.append(line.rstrip())
    logger.info(f'{len(sentences)} sentences loaded')
    logger.debug(f'Loaded text\n\n{sentences}')

    start_time = time()
    logger.info(f'Loading IDF corpus \'{args.idf}\'')
    documents = list()
    for filepath in args.idf.iterdir():
        with filepath.open('rU', encoding='utf-8') as file_:
            documents.append(file_.readlines())
    end_time = time()
    logger.info(
        f'Loaded {len(documents)} documents after {end_time - start_time}s')

    logger.info('Summarizing')
    lxr = LexRank(documents, stopwords=STOPWORDS['pt'])
    logger.debug(f'Stopwords used\n\n{lxr.stopwords}')
    summary_size = ceil(args.ratio * len(sentences)) + 1
    summary = lxr.get_summary(sentences,
                              summary_size=summary_size)
    logger.info('Summarization done')
    logger.debug(f'Summary\n\n{summary}')
    logger.info(f'Saving summary with {summary_size} sentences')
    with args.output.open('w', encoding='utf-8') as file_:
        for sent in summary:
            file_.write(f'{sent}\n')
    logger.info(f'Saved summary to \'{args.output}\'')


def summarize_bert(args):
    logger.info(f'Loading file \'{args.file}\'')
    sentences = list()
    with args.file.open('rU', encoding='utf-8') as file_:
        sent_re = re.compile('(\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)')
        for match in sent_re.finditer(file_.read()):
            sent = match.group()
            if sent not in sentences:
                sentences.append(sent)
    logger.info(f'{len(sentences)} sentences loaded')
    logger.debug(f'Loaded text\n\n{sentences}')

    logger.info('Loading BERT model')
    model = SentenceTransformer(args.model)
    logger.info('Embedding sentences')
    embeddings = model.encode(sentences)
    logger.debug(f'Embeddings shape: {embeddings.shape}')

    logger.info('Calculating similarity between sentences')
    similarity = util.pytorch_cos_sim(embeddings, embeddings).numpy()
    logger.debug(f'Similarity matrix\n\n{similarity}')
    logger.debug(f'Similarity shape: {similarity.shape}')

    logger.info('Summarizing')
    row_sum = similarity.sum(axis=1, keepdims=True)
    markov_matrix = similarity / row_sum
    logger.debug(f'Markov matrix\n\n{markov_matrix}')
    logger.debug(f'Markov matrix shape: {markov_matrix.shape}')
    scores = stationary_distribution(markov_matrix,
                                     increase_power=True,
                                     normalized=False)
    logger.debug(f'Centrality values\n\n{scores}')
    summary_size = ceil(args.ratio * len(sentences)) + 1
    sorted_ix = np.argsort(scores)[::-1][:summary_size]
    sorted_ix = sorted(sorted_ix)
    logger.debug(f'Summary indices: {sorted_ix}')
    summary = [sentences[i] for i in sorted_ix]
    logger.debug(f'Summary\n\n{summary}')
    logger.info(f'Saving summary with {summary_size} sentences')
    with args.output.open('w', encoding='utf-8') as file_:
        for sent in summary:
            file_.write(f'{sent}\n')
    logger.info(f'Saved summary to \'{args.output}\'')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f',
                        help='Input file to be summarized',
                        type=Path, required=True)
    parser.add_argument('--method', '-m',
                        help='Method to use for similarity measure',
                        choices=['tf-idf', 'bert'],
                        required=True)
    parser.add_argument('--output', '-o',
                        help='File to save the resulting summary',
                        type=Path, required=True)
    parser.add_argument('--idf', '-i',
                        help='Directory with documents upon which calculate IDF',
                        type=Path, required=False)
    parser.add_argument('--model', '-M',
                        help='BERT model', type=str,
                        default='ricardo-filho/bert-portuguese-cased-nli-assin-assin-2',
                        required=False)
    parser.add_argument('--language', '-l',
                        help='Language for summarization',
                        type=str, default='portuguese',
                        required=False)
    parser.add_argument('--ratio', '-r',
                        help='Compression rate',
                        type=float, default=0.2,
                        required=False)
    parser.add_argument('--verbose', '-v',
                        action='count', default=0,
                        help='Print information and debugging messages',
                        required=False)
    args = parser.parse_args()

    ch = logging.StreamHandler()
    if args.verbose == 1:
        logger.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
    elif args.verbose >= 1:
        logger.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if args.method == 'tf-idf':
        summarize_idf(args)
    elif args.method == 'bert':
        summarize_bert(args)
