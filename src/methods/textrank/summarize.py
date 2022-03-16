import argparse
import logging
from pathlib import Path
from summa import summarizer


def summarize(args):
    logging.info(f'Loading file \'{args.file}\'')
    with args.file.open('rU', encoding='utf-8') as file_:
        text = file_.read()
    logging.info(f'File loaded')
    logging.debug(f'Loaded text\n\n{text}')

    logging.info('Summarizing')
    summary = summarizer.summarize(text,
                                   language=args.language,
                                   ratio=args.ratio)
    logging.info('Summarization done')
    logging.debug(f'Summary\n\n{summary}')

    logging.info('Saving summary')
    with args.output.open('w', encoding='utf-8') as file_:
        file_.write(summary)
    logging.info(f'Saved summary to \'{args.output}\'')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f',
                        help='Input file to be summarized',
                        type=Path, required=True)
    parser.add_argument('--output', '-o',
                        help='File to save the resulting summary',
                        type=Path, required=True)
    parser.add_argument('--language', '-l',
                        help='Language for summarization',
                        type=str, default='portuguese')
    parser.add_argument('--ratio', '-r',
                        help='Compression rate',
                        type=float, default=0.2)
    parser.add_argument('--verbose', '-v',
                        action='count', default=0,
                        help='Print information and debugging messages')
    args = parser.parse_args()

    if args.verbose == 1:
        logging.basicConfig(level='INFO')
    elif args.verbose >= 2:
        logging.basicConfig(level='DEBUG')

    summarize(args)
