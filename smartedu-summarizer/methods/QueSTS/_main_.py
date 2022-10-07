import argparse
import logging
from pathlib import Path

from ..tf_idf.summarize import summarize

logger = logging.getLogger('smartedu-summarizer')

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f',
                    help='Input file to be summarized',
                    type=Path, required=True)
parser.add_argument('--output', '-o',
                    help='File to save the resulting summary',
                    type=Path, required=True)
parser.add_argument('--language', '-l',
                    help='Language for summarization',
                    type=str, default='english')
parser.add_argument('--verbose', '-v',
                    action='count', default=0,
                    help='Print information and debugging messages')
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

summarize(args)