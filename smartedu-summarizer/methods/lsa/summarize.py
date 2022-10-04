import logging
from math import ceil
import nltk

from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

logger = logging.getLogger('smartedu-summarizer')


def sentence_split(paragraph):
    return nltk.sent_tokenize(paragraph)

def summarize(args):
    logger.info(f'Loading file \'{args.file}\'')
    with args.file.open('rU', encoding='utf-8') as file_:
        text = file_.read()
    logger.info(f'File loaded')
    logger.debug(f'Loaded text\n\n{text}')

    logger.info('Summarizing')
    sentences = sentence_split(text)
    summary_size = ceil(args.ratio * len(sentences)) + 1

    parser = PlaintextParser.from_string(text, Tokenizer(args.language))

    # creating the LSA summarizer 
    lsa_summarizer = LsaSummarizer()
    lsa_summary = lsa_summarizer(parser.document, summary_size)

    summary = ''    
    for sentence in lsa_summary:
        summary += str(sentence) + " "

    logger.info('Summarization done')
    logger.debug(f'Summary\n\n{summary}')

    logger.info('Saving summary')
    with args.output.open('w', encoding='utf-8') as file_:
        file_.write(summary)
    logger.info(f'Saved summary to \'{args.output}\'')