import argparse
import logging
from pathlib import Path

from ..lexrank.summarize import summarize_bert, summarize_idf

logger = logging.getLogger('smartedu-summarizer')

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
parser.add_argument('--num_docs', '-n',
                    help='Number of documents upon which calculate IDF',
                    type=int, default=None)
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
    ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

if args.method == 'tf-idf':
    summarize_idf(args.file, args.idf, args.ratio, args.num_docs, args.output)
elif args.method == 'bert':
    summarize_bert(args.file, args.model, args.ratio, args.output)
