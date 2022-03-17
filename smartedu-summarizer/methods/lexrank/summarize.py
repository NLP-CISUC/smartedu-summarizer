import logging
import re
from math import ceil
from pathlib import Path
from time import time
from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer, util

from lexrank import STOPWORDS, LexRank
from lexrank.algorithms.power_method import stationary_distribution

logger = logging.getLogger('smartedu-summarizer')


def summarize_idf(filepath: Path, idfpath: Path,
                  ratio: float, num_docs: Union[int, None],
                  outpath: Path) -> None:
    logger.info(f'Loading file \'{filepath}\'')
    sentences = list()
    with filepath.open('rU', encoding='utf-8') as file_:
        sent_re = re.compile('(\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)')
        for match in sent_re.finditer(file_.read()):
            sent = match.group()
            if sent not in sentences:
                sentences.append(sent)
    logger.info(f'{len(sentences)} sentences loaded')
    logger.debug(f'Loaded text\n\n{sentences}')

    start_time = time()
    logger.info(f'Loading IDF corpus \'{idfpath}\'')
    documents = list()
    for i, filepath in enumerate(idfpath.iterdir(), start=1):
        with filepath.open('rU', encoding='utf-8') as file_:
            documents.append(file_.readlines())

            if num_docs is not None and i == num_docs:
                break
    end_time = time()
    lxr = LexRank(documents, stopwords=STOPWORDS['pt'])
    logger.info(
        f'Loaded {len(documents)} documents after {end_time - start_time}s')

    logger.info('Summarizing')
    logger.debug(f'Stopwords used\n\n{lxr.stopwords}')
    summary_size = ceil(ratio * len(sentences)) + 1
    summary = lxr.get_summary(sentences,
                              summary_size=summary_size)
    logging.info('Sort by apparition in original text')
    summary.sort(key=sentences.index)
    logger.info('Summarization done')
    logger.debug(f'Summary\n\n{summary}')
    logger.info(f'Saving summary with {summary_size} sentences')
    with outpath.open('w', encoding='utf-8') as file_:
        for sent in summary:
            file_.write(f'{sent}\n')
    logger.info(f'Saved summary to \'{outpath}\'')


def summarize_bert(filepath: Path, model: str,
                   ratio: float, outpath: Path):
    logger.info(f'Loading file \'{filepath}\'')
    sentences = list()
    with filepath.open('rU', encoding='utf-8') as file_:
        sent_re = re.compile('(\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)')
        for match in sent_re.finditer(file_.read()):
            sent = match.group()
            if sent not in sentences:
                sentences.append(sent)
    logger.info(f'{len(sentences)} sentences loaded')
    logger.debug(f'Loaded text\n\n{sentences}')

    logger.info('Loading BERT model')
    model = SentenceTransformer(model)
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
    summary_size = ceil(ratio * len(sentences)) + 1
    sorted_ix = np.argsort(scores)[::-1][:summary_size]
    sorted_ix = sorted(sorted_ix)
    logger.debug(f'Summary indices: {sorted_ix}')
    summary = [sentences[i] for i in sorted_ix]
    logger.debug(f'Summary\n\n{summary}')
    logger.info(f'Saving summary with {summary_size} sentences')
    with outpath.open('w', encoding='utf-8') as file_:
        for sent in summary:
            file_.write(f'{sent}\n')
    logger.info(f'Saved summary to \'{outpath}\'')
