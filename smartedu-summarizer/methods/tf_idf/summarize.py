import logging
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

logger = logging.getLogger('smartedu-summarizer')


def sentence_split(paragraph):
    return nltk.sent_tokenize(paragraph)


def sentence_score(sentence_list, df):
    sentences_scores = [0] * len(sentence_list)

    for sentence in range(len(sentence_list)):
        count_words = 0
        sum_values = 0
        for value in range(len(df.iloc[[0]].values[0])):
            num = df.iloc[[sentence]].values[0][value]
            if (num != 0):
                count_words += 1
                sum_values += num
        if count_words != 0:
            sentences_scores[sentence] =  sum_values/count_words

    return sentences_scores


def TF_IDF(text, language):
    
    #sentences are docs
    sentence_list = sentence_split(text)

    stopWords = set(stopwords.words(language))

    vectorizer = TfidfVectorizer(stop_words=stopWords)
    vectors = vectorizer.fit_transform(sentence_list)
    feature_names = vectorizer.get_feature_names()

    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)

    sentences_scores = sentence_score(sentence_list, df)
    average_scores = sum(sentences_scores) / len(sentences_scores)

    summary = []
    for i in range(len(sentences_scores)):
        if sentences_scores[i] > average_scores:
            summary.append(sentence_list[i])

    result = ''
    for i in range(len(summary)):
        result += '{} '.format(summary[i])

    return result


def summarize(args):
    logger.info(f'Loading file \'{args.file}\'')
    with args.file.open('rU', encoding='utf-8') as file_:
        text = file_.read()
    logger.info(f'File loaded')
    logger.debug(f'Loaded text\n\n{text}')

    logger.info('Summarizing')
    summary = TF_IDF(text, args.language)
    logger.info('Summarization done')
    logger.debug(f'Summary\n\n{summary}')

    logger.info('Saving summary')
    with args.output.open('w', encoding='utf-8') as file_:
        file_.write(summary)
    logger.info(f'Saved summary to \'{args.output}\'')