import logging
from pathlib import Path

from summa import summarizer

logger = logging.getLogger('smartedu-summarizer')


def summarize(args):
    logger.info(f'Loading file \'{args.file}\'')
    with args.file.open('rU', encoding='utf-8') as file_:
        text = file_.read()
    logger.info(f'File loaded')
    logger.debug(f'Loaded text\n\n{text}')

    logger.info('Summarizing')
    summary = summarizer.summarize(text,
                                   language=args.language,
                                   ratio=args.ratio)
    logger.info('Summarization done')
    logger.debug(f'Summary\n\n{summary}')

    logger.info('Saving summary')
    with args.output.open('w', encoding='utf-8') as file_:
        file_.write(summary)
    logger.info(f'Saved summary to \'{args.output}\'')
